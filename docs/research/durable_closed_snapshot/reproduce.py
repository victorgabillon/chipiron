"""Run durable measurements serially, with a fresh interpreter for every stage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run one explicitly requested synthetic measurement."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--history", action="store_true")
    p.add_argument("--last", type=int, default=64)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "docs.research.durable_closed_snapshot.measure",
        str(args.root),
        "--count",
        str(args.count),
    ]
    if args.history:
        command.append("--history")
    jobs = ["initial"]
    if args.history:
        for g in range(2, args.last + 1):
            jobs.append("repeat")
            if g in (8, 16, 32, 64):
                jobs += ["discover", "cold", "reuse", "oracle"]
    else:
        jobs += [
            "repeat",
            "repeat",
            "discover",
            "cold",
            "reuse",
            "oracle",
            "dataset",
            "continuation",
        ]
    completed = []
    for i, job in enumerate(jobs):
        print(json.dumps({"stage": i, "mode": job, "count": args.count}), flush=True)
        with (args.root / f"stage-{i:03d}-{job}.log").open("w") as log:
            result = subprocess.run(
                [*command, job], stdout=log, stderr=subprocess.STDOUT
            )
        completed.append({"command": [*command, job], "returncode": result.returncode})
        (args.root / "commands.json").write_text(json.dumps(completed, indent=2) + "\n")
        if result.returncode:
            message = f"Failed {job} at {i}; see log"
            raise SystemExit(message)
    print(json.dumps({"completed": len(completed), "root": str(args.root)}), flush=True)


if __name__ == "__main__":
    main()
