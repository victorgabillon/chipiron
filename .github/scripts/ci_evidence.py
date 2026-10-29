"""Bind each runtime artifact to its lane, immutable candidate and CI attempt."""

import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    """Record attribution before tests, including the candidate's test tree."""
    candidate = os.environ["GITHUB_SHA"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != candidate:
        message = f"Evidence candidate {candidate} differs from checkout {head}"
        raise RuntimeError(message)
    identities = {
        path: subprocess.check_output(
            ["git", "rev-parse", f"{candidate}:{path}"], text=True
        ).strip()
        for path in (
            "tests",
            "src/chipiron",
            "Makefile",
            "pyproject.toml",
            ".github/ci-source-contracts.json",
            ".github/ci-cpu-constraints.txt",
            ".github/workflows/ci.yaml",
        )
    }
    output = Path(".ci-reports")
    output.mkdir(exist_ok=True)
    (output / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": candidate,
                "lane": os.environ["CI_TEST_LANE"],
                "run_id": os.environ["GITHUB_RUN_ID"],
                "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                "event": os.environ["GITHUB_EVENT_NAME"],
                "git_objects": identities,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
