from pathlib import Path
import argparse

from chipiron.environments.morpion.bootstrap.dashboard_app import run_dashboard_app

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()

run_dashboard_app(args.work_dir)
