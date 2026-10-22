"""Run one requested size serially; every measured stage gets a fresh process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run main for the isolated synthetic study."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "docs.research.closed_scientific_snapshot.measure",
        str(args.root),
        "--count",
        str(args.count),
        "--offset",
        str(args.offset),
    ]
    jobs = [
        [mode]
        for mode in (
            "prepare",
            "initial",
            "repeat",
            "growth",
            "cold",
            "reuse",
            "reference",
        )
    ]
    jobs += [[mode, "--snapshot", "growth"] for mode in ("cold", "reference")]
    if args.legacy:
        jobs.append(["dataset", "--policy", "all", "--confidence", "legacy_v1"])
    if args.datasets:
        jobs.append(["dataset", "--policy", "all"])
    completed = []
    for index, job in enumerate(jobs):
        print(
            json.dumps({"starting": job, "count": args.count, "offset": args.offset}),
            flush=True,
        )
        with (args.root / f"stage-{index:02d}.log").open("w") as log:
            result = subprocess.run(
                [*command, *job], stdout=log, stderr=subprocess.STDOUT, check=False
            )
        completed.append({"command": [*command, *job], "returncode": result.returncode})
        (args.root / "commands.json").write_text(json.dumps(completed, indent=2) + "\n")
        if result.returncode:
            message = f"failed stage {index}: {job}; see log"
            raise SystemExit(message)
    print(json.dumps({"completed": len(completed), "root": str(args.root)}), flush=True)


if __name__ == "__main__":
    main()
