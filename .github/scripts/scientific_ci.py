"""Prepare pinned scientific dependencies and verify installed package contents."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(root: Path, expected: dict) -> None:
    """Reject a missing or changed source member."""
    for name, wanted in expected.items():
        if digest(root / name) != wanted:
            message = f"Scientific source mismatch: {name}"
            raise RuntimeError(message)


def prepare(contract: dict) -> None:
    """Build the immutable historical dependency outside the source checkout."""
    source = contract.get("source_build")
    if source is None:
        return
    temporary = Path(os.environ["RUNNER_TEMP"]) / "chipiron-scientific-dependency"
    temporary.mkdir()
    archive = temporary / "source.tar.gz"
    with urllib.request.urlopen(source["url"], timeout=120) as response:
        archive.write_bytes(response.read())
    with tarfile.open(archive) as bundle:
        bundle.extractall(temporary / "source", filter="data")
    checkout = next((temporary / "source").iterdir())
    check_files(checkout, source["files"])
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *source["build_deps"]], check=True
    )
    wheels = temporary / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(checkout),
        ],
        check=True,
    )
    wheel = next(wheels.glob("*.whl"))
    constraint = temporary / "constraints.txt"
    constraint.write_text(
        (ROOT / ".github/ci-historical-constraints.txt").read_text()
        + f"\nalgorhino-anemone @ {wheel.as_uri()}#sha256={digest(wheel)}\n"
    )
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"PIP_CONSTRAINT={constraint}\n")
    (ROOT / ".ci-reports").mkdir(exist_ok=True)
    (ROOT / ".ci-reports/scientific-dependency-build.json").write_text(
        json.dumps(
            {"commit": source["commit"], "wheel": wheel.name, "sha256": digest(wheel)},
            indent=2,
        )
        + "\n"
    )


def immutable_commit(reference: str, *, require_head: bool = False) -> str:
    """Require a full commit identity, never a branch, tag, tree or abbreviated SHA."""
    if re.fullmatch(r"[0-9a-f]{40}", reference) is None:
        message = "Scientific source reference must be a full immutable commit SHA"
        raise RuntimeError(message)
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{reference}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    if resolved != reference:
        message = f"Scientific source reference is not a commit: {reference}"
        raise RuntimeError(message)
    if require_head:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if head != reference:
            message = f"Build candidate {reference} differs from checkout HEAD {head}"
            raise RuntimeError(message)
    return reference


def is_source_member(path: Path) -> bool:
    """Include Python typing sources and reject unexpected native implementations."""
    return (
        path.suffix in {".py", ".pyi", ".pyc", ".so", ".pyd"} or path.name == "py.typed"
    )


def git_members(reference: str, *paths: str) -> dict[str, str]:
    """Read tracked paths and their immutable Git blob identities."""
    entries = (
        subprocess
        .check_output(["git", "ls-tree", "-rz", reference, "--", *paths], cwd=ROOT)
        .decode()
        .split("\0")
    )
    result = {}
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        result[name] = metadata.split()[2]
    return result


def git_text(reference: str, name: str) -> str:
    """Read build declarations from the candidate, never the mutable checkout."""
    return subprocess.check_output(
        ["git", "show", f"{reference}:{name}"], cwd=ROOT, text=True
    )


def check_git_bytes(root: Path, expected: dict[str, str]) -> None:
    """Reject source overrides and changed or missing committed payload bytes."""
    for name, wanted in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            message = f"Chipiron source override: {path}"
            raise RuntimeError(message)
        data = path.read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if blob != wanted:
            message = f"Chipiron differs from reviewed Git source: {path}"
            raise RuntimeError(message)


def verify_build_inputs(reference: str) -> None:
    """Bind packaging declarations and scientific dependency identities to the build."""
    inputs = (
        "pyproject.toml",
        "MANIFEST.in",
        "setup.py",
        "setup.cfg",
        ".github/ci-source-contracts.json",
    )
    expected = git_members(reference, *inputs)
    actual = {name for name in inputs if (ROOT / name).exists()}
    if actual != set(expected):
        message = "Chipiron build input membership differs from reviewed Git source"
        raise RuntimeError(message)
    check_git_bytes(ROOT, expected)


def package_rules(reference: str) -> tuple[dict, list[list[str]]]:
    """Read the repository's explicit payload declarations, rejecting unknown rules."""
    config = tomllib.loads(git_text(reference, "pyproject.toml"))
    setuptools = config["tool"]["setuptools"]
    if setuptools.get("exclude-package-data") or not setuptools.get(
        "include-package-data", True
    ):
        message = "Unsupported Chipiron package-data configuration"
        raise RuntimeError(message)
    for package in setuptools["package-data"]:
        if not package or "*" in package:
            message = f"Unsupported Chipiron package-data package: {package}"
            raise RuntimeError(message)
    manifest = []
    if git_members(reference, "MANIFEST.in"):
        for line in git_text(reference, "MANIFEST.in").splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            command, *args = tokens
            valid = (
                (command == "recursive-include" and len(args) >= 2)
                or (command in {"include", "global-exclude"} and bool(args))
                or (command == "prune" and len(args) == 1)
            )
            if not valid or (
                command == "recursive-include" and any(c in args[0] for c in "*?[")
            ):
                message = f"Unsupported Chipiron manifest rule: {line.strip()}"
                raise RuntimeError(message)
            manifest.append(tokens)
    return config, manifest


def is_package_member(name: str, config: dict, manifest: list[list[str]]) -> bool:
    """Select package payload using immutable package-data and MANIFEST declarations."""
    path = PurePosixPath(name)
    selected = is_source_member(Path(name))
    for package, patterns in config["tool"]["setuptools"]["package-data"].items():
        prefix = package.replace(".", "/")
        if path.is_relative_to(prefix):
            relative = path.relative_to(prefix)
            selected = selected or any(relative.full_match(p) for p in patterns)
    source_path = PurePosixPath("src") / path
    manifest_selected = False
    for command, *args in manifest:
        if command == "recursive-include":
            directory, *patterns = args
            if source_path.is_relative_to(directory):
                manifest_selected = manifest_selected or any(
                    PurePosixPath(source_path.name).full_match(p) for p in patterns
                )
        elif command == "include":
            manifest_selected = manifest_selected or any(
                source_path.full_match(p) for p in args
            )
        elif command == "prune":
            if any(parent.full_match(args[0]) for parent in source_path.parents):
                manifest_selected = False
        elif any(PurePosixPath(source_path.name).full_match(p) for p in args):
            manifest_selected = False
    return selected or manifest_selected


def verify_chipiron_sources(package_root: Path, reference: str) -> tuple[int, int]:
    """Check complete installed payload plus the checkout's declared package members."""
    config, manifest = package_rules(reference)
    expected = {
        name.removeprefix("src/"): blob
        for name, blob in git_members(reference, "src/chipiron").items()
        if is_package_member(name.removeprefix("src/"), config, manifest)
    }
    if "chipiron/__init__.py" not in expected:
        message = f"No Chipiron sources in reference: {reference}"
        raise RuntimeError(message)
    for root in dict.fromkeys((package_root, ROOT / "src")):
        checkout = root == ROOT / "src"
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / "chipiron").rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (
                not checkout
                or is_source_member(path)
                or is_package_member(
                    path.relative_to(root).as_posix(), config, manifest
                )
            )
        }
        if actual != set(expected):
            message = (
                f"Chipiron source membership mismatch at {root}: "
                f"missing={sorted(set(expected) - actual)}, "
                f"unexpected={sorted(actual - set(expected))}"
            )
            raise RuntimeError(message)
        check_git_bytes(root, expected)
    return sum(is_source_member(Path(name)) for name in expected), len(expected)


def verify(
    contract: dict,
    *,
    editable_chipiron: bool = False,
    candidate: str | None = None,
) -> None:
    """Require exact installed scientific dependencies and current Chipiron source."""
    reference = immutable_commit(
        contract["chipiron_reference"] if candidate is None else candidate,
        require_head=candidate is not None,
    )
    if candidate is not None:
        verify_build_inputs(reference)
    site = Path(sysconfig.get_path("purelib")).resolve()
    result = {}
    # Frozen replay keeps the historical map. Candidate overrides replace whole
    # package identities, including membership, without mutating that map.
    packages = contract["packages"]
    if candidate is not None:
        packages = packages | contract.get("candidate_package_overrides", {})
    for distribution, package in packages.items():
        if importlib.metadata.version(distribution) != package["version"]:
            message = f"Wrong scientific dependency version: {distribution}"
            raise RuntimeError(message)
        spec = importlib.util.find_spec(package["module"])
        if spec is None or spec.origin is None:
            message = f"Missing installed package: {distribution}"
            raise RuntimeError(message)
        origin = Path(spec.origin).resolve()
        if origin != site / package["module"] / "__init__.py":
            message = f"Source override: {origin}"
            raise RuntimeError(message)
        check_files(site, package["files"])
        actual = {
            str(path.relative_to(site))
            for path in (site / package["module"]).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if actual != set(package["files"]):
            message = f"Unexpected scientific package members: {distribution}"
            raise RuntimeError(message)
        result[distribution] = {"origin": str(origin), "members": len(actual)}
    package_root = ROOT / "src" if editable_chipiron else site
    spec = importlib.util.find_spec("chipiron")
    expected_origin = package_root / "chipiron/__init__.py"
    if (
        spec is None
        or spec.origin is None
        or Path(spec.origin).resolve() != expected_origin.resolve()
    ):
        message = f"Unexpected Chipiron origin; expected {expected_origin}"
        raise RuntimeError(message)
    source_members, package_members = verify_chipiron_sources(package_root, reference)
    result["chipiron"] = {
        "origin": str(expected_origin),
        "editable": editable_chipiron,
        "reference": reference,
        "mode": "frozen-replay" if candidate is None else "build-candidate",
        "source_members": source_members,
        "package_members": package_members,
    }
    (ROOT / ".ci-reports").mkdir(exist_ok=True)
    (ROOT / ".ci-reports/scientific-installed-identity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    """Validate the exact interpreter before preparing or inspecting packages."""
    contract = json.loads((ROOT / ".github/ci-source-contracts.json").read_text())
    if platform.python_version() != contract["python"]:
        message = f"Expected Python {contract['python']}"
        raise RuntimeError(message)
    if sys.argv[1:] == ["prepare"]:
        prepare(contract)
    elif sys.argv[1:] == ["verify"]:
        verify(contract)
    elif sys.argv[1:] == ["verify-static"]:
        verify(contract, editable_chipiron=True)
    elif len(sys.argv) == 3 and sys.argv[1] in (
        "verify-candidate",
        "verify-static-candidate",
    ):
        verify(
            contract,
            editable_chipiron=sys.argv[1] == "verify-static-candidate",
            candidate=sys.argv[2],
        )
    else:
        message = (
            "Usage: scientific_ci.py prepare|verify|verify-static "
            "or verify-candidate|verify-static-candidate FULL_COMMIT_SHA"
        )
        raise SystemExit(message)


if __name__ == "__main__":
    main()
