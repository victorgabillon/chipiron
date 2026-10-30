"""Read-only post-run report using the same evidence and timeline as Operations.

Usage: python -m chipiron.environments.morpion.bootstrap.performance_report --work-dir PATH
Redirect stdout to a runtime directory to retain JSON. No workspace files are written.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .dashboard.performance_view import load_performance_summary

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Write a JSON summary to stdout, without repairing or loading scientific artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            asdict(load_performance_summary(args.work_dir)), indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
