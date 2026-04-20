from __future__ import annotations

import argparse
from pathlib import Path

from chipiron.environments.morpion.bootstrap.dashboard_app import run_dashboard_app

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    run_dashboard_app(Path(args.work_dir))

if __name__ == "__main__":
    main()