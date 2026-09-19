"""Record runtime dependency provenance without importing heavy model packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PACKAGES = {
    "chipiron": "chipiron",
    "algorhino-coral": "coral",
    "algorhino-anemone": "anemone",
    "parsley-coco": "parsley",
    "atomheart": "atomheart",
    "valanga": "valanga",
    "torch": "torch",
}


def git_output(root: Path, *args: str) -> str | None:
    """Read Git metadata without altering a repository or its index."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        return None
    return result.stdout if "-z" in args else result.stdout.strip()


def git_state(path: Path) -> dict[str, object] | None:
    """Describe the actual imported checkout, including uncommitted-file hashes."""
    root_text = git_output(
        path if path.is_dir() else path.parent, "rev-parse", "--show-toplevel"
    )
    if root_text is None:
        return None
    root = Path(root_text)
    status = git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    changed = git_output(
        root, "ls-files", "--modified", "--others", "--exclude-standard", "-z"
    )
    checksums = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (changed or "").split("\0")
        if name and (root / name).is_file()
    }
    return {
        "root": str(root),
        "commit": git_output(root, "rev-parse", "HEAD"),
        "branch": git_output(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status,
        "changed_files_sha256": checksums,
    }


def sanitized_direct_url(raw: str) -> dict[str, object]:
    """Keep immutable installer identity without URL credentials/query tokens."""
    record = json.loads(raw)
    url = urlsplit(str(record.get("url", "")))
    host = url.hostname or ""
    if url.port is not None:
        host += f":{url.port}"
    record["url"] = urlunsplit((url.scheme, host, url.path, "", ""))
    return record


def dependency_state(distribution_name: str, module_name: str) -> dict[str, object]:
    """Record installed version and the source actually selected by Python."""
    result: dict[str, object] = {"distribution": distribution_name}
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = None
    else:
        result["version"] = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            result["direct_url"] = sanitized_direct_url(direct_url)
        installed_record = distribution.read_text("RECORD")
        if installed_record:
            result["installed_record_sha256"] = hashlib.sha256(
                installed_record.encode()
            ).hexdigest()
    spec = importlib.util.find_spec(module_name)
    origin = None if spec is None else spec.origin
    result["import_origin"] = origin
    if origin is not None:
        result["source_git"] = git_state(Path(origin))
    return result


def collect_dependency_state() -> dict[str, object]:
    """Build a JSON-compatible record suitable for existing experiment metadata."""
    return {
        "schema": "chipiron_dependency_state_v1",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "chipiron_checkout": git_state(Path(__file__).resolve().parents[1]),
        "packages": {
            name: dependency_state(name, module) for name, module in PACKAGES.items()
        },
        "limitations": "Dirty checkouts require an external source/patch archive; hashes alone do not reconstruct edits. Model/data hashes belong in the experiment's existing manifest.",
    }


def has_unpinned_sources(record: dict[str, object]) -> bool:
    """Reject missing packages, dirty checkouts and VCS installs without commit IDs."""
    packages = record["packages"]
    assert isinstance(packages, dict)
    checkout = record["chipiron_checkout"]
    if isinstance(checkout, dict) and checkout.get("dirty"):
        return True
    for package in packages.values():
        if package.get("version") is None or package.get("import_origin") is None:
            return True
        source = package.get("source_git")
        if isinstance(source, dict) and (
            source.get("dirty") or not source.get("commit")
        ):
            return True
        direct = package.get("direct_url", {})
        vcs = direct.get("vcs_info")
        if vcs is not None and not vcs.get("commit_id"):
            return True
    return False


def main() -> int:
    """Write provenance beside external artifacts, optionally requiring clean sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    record = collect_dependency_state()
    record["clean_immutable_sources"] = not has_unpinned_sources(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Dependency state written to {args.output}")
    return int(args.require_clean and not record["clean_immutable_sources"])


if __name__ == "__main__":
    raise SystemExit(main())
