"""Collect completed benchmark JSON and derive review tables without rereading trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Run main for the isolated synthetic study."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = {}
    for directory in sorted(args.evidence.glob("final-*")):
        if not directory.is_dir():
            continue
        records = {
            p.stem: json.loads(p.read_text())
            for p in directory.glob("*.json")
            if p.stem != "commands"
        }
        commands = json.loads((directory / "commands.json").read_text())
        assert all(item["returncode"] == 0 for item in commands)
        cases[directory.name] = records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + "\n")
    for name, case in cases.items():
        print(name)
        for mode in ("initial", "repeat", "growth", "cold", "reuse", "reference"):
            result = case[mode]
            observation = result["measurement"]
            print(
                mode,
                round(observation["elapsed_s"], 3),
                round(observation["process_peak_rss_mb"], 1),
                observation["counters"]["state_decodes"],
            )
        print(
            "JSONL rows",
            sum(
                row["exact_jsonl_rows"]
                for key, result in case.items()
                if key.startswith("dataset-")
                for row in result["comparisons"]
            ),
        )


if __name__ == "__main__":
    main()
