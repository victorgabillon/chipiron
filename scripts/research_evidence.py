"""Verify or restore archived research outputs using only Git and the stdlib."""

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/research/evidence-manifest.csv"


def restore(destination: Path | None = None) -> tuple[int, int]:
    """Verify every original blob; optionally restore to a new directory."""
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    count = total = 0
    seen: set[str] = set()
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["path"]
            path = PurePosixPath(name)
            commit, blob = row["source_commit"], row["git_blob"]
            if (
                path.is_absolute()
                or ".." in path.parts
                or not name.startswith("docs/research/")
                or "/evidence/" not in name
                or name in seen
                or not re.fullmatch(r"[0-9a-f]{40}", commit)
                or not re.fullmatch(r"[0-9a-f]{40}", blob)
            ):
                message = f"Invalid manifest entry: {name}"
                raise ValueError(message)
            seen.add(name)
            actual_blob = subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{name}"], cwd=ROOT, text=True
            ).strip()
            if actual_blob != blob:
                message = f"Git identity mismatch: {name}"
                raise ValueError(message)
            content = subprocess.check_output(
                ["git", "cat-file", "blob", blob], cwd=ROOT
            )
            if (
                len(content) != int(row["bytes"])
                or hashlib.sha256(content).hexdigest() != row["sha256"]
            ):
                message = f"Content mismatch: {name}"
                raise ValueError(message)
            if destination is not None:
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(content)
                if target.read_bytes() != content:
                    message = f"Restoration mismatch: {name}"
                    raise ValueError(message)
            count += 1
            total += len(content)
    return count, total


def main() -> None:
    """Verify the manifest, or restore it without overwriting existing data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="new restoration directory")
    options = parser.parse_args()
    count, total = restore(options.destination)
    print(f"Verified {count} files / {total} bytes (Git blob, size, SHA-256).")


if __name__ == "__main__":
    main()
