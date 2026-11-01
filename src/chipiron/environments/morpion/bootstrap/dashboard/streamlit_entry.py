"""CLI entrypoint for the Morpion bootstrap Streamlit dashboard."""

import argparse
from pathlib import Path

from chipiron.environments.morpion.bootstrap.dashboard import run_dashboard_app


def main() -> None:
    """Parse CLI args and launch the Streamlit dashboard app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="Optional explicitly associated supervisor/runtime directory",
    )
    args = parser.parse_args()
    run_dashboard_app(Path(args.work_dir), args.runtime_root)


if __name__ == "__main__":
    main()
