"""Small baseline probe; reuse existing stage/RSS/disk instruments verbatim."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chipiron.environments.morpion.bootstrap.pipeline import stages
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.fenced_dataset.test_baseline import adapter
from tests.environments.morpion_atomic_reevaluation_helpers import setup_atomic


def main() -> None:
    """Measure one freshly generated closed-source Dataset in one process."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", choices=["legacy", "fenced"], default="legacy")
    opt = parser.parse_args()
    root = opt.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if opt.protocol == "fenced":
        from tests.environments.morpion_fenced_dataset_helpers import setup_fenced

        args, paths, _ = setup_fenced(root / "work")
    else:
        args, paths, _ = setup_atomic(root / "work")
    args = replace(args, use_backed_up_value=False)
    instruments.options = SimpleNamespace(profile=False)
    with pytest.MonkeyPatch.context() as patch:
        adapter(patch, paths.work_dir)
        result, observation = instruments.measured(
            "dataset",
            root,
            lambda: stages.run_pipeline_dataset_stage(
                args,
                generation=1,
                use_streaming_sharded_reader=True,
                allow_scientific_v2=True,
                reader_cache_dir=root / "cache",
            ),
        )
    rows = paths.work_dir / result.rows_path
    observation["rows_sha256"] = hashlib.sha256(rows.read_bytes()).hexdigest()
    observation["rows_bytes"] = rows.stat().st_size
    observation["rows"] = result.metadata["dataset_rows"]
    observation["protocol"] = opt.protocol
    observation["run_state_bytes"] = paths.run_state_path.stat().st_size
    (root / "measurement.json").write_text(json.dumps(observation, indent=2) + "\n")


if __name__ == "__main__":
    main()
