"""Prepare pinned scientific dependencies and verify installed package contents."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

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


def verify(contract: dict) -> None:
    """Require exact installed scientific dependencies and current Chipiron source."""
    site = Path(sysconfig.get_path("purelib")).resolve()
    result = {}
    for distribution, package in contract["packages"].items():
        if importlib.metadata.version(distribution) != package["version"]:
            message = f"Wrong scientific dependency version: {distribution}"
            raise RuntimeError(message)
        spec = importlib.util.find_spec(package["module"])
        if spec is None or spec.origin is None:
            message = f"Missing installed package: {distribution}"
            raise RuntimeError(message)
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(site):
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
    sources = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "src/chipiron"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for name in sources:
        if (
            name.endswith((".py", ".pyi", "/py.typed"))
            and (ROOT / name).read_bytes()
            != (site / name.removeprefix("src/")).read_bytes()
        ):
            message = f"Installed Chipiron differs from reviewed source: {name}"
            raise RuntimeError(message)
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
    else:
        message = "Usage: scientific_ci.py prepare|verify"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
