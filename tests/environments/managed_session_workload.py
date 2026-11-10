"""Fixed tiny scientific work inside real monitored ancestry; validation fixture only."""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.execution_profile import (
    bootstrap_execution,
)
from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow


def main() -> None:
    """Execute fixed accepted rounds through existing stages, then await native stop."""
    root = Path(sys.argv[1])
    paths = MorpionBootstrapPaths.from_work_dir(root)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    args = replace(
        _scientific_args(MorpionBootstrapArgs(work_dir=root), config),
        max_growth_steps_per_cycle=2,
        max_rows=128,
    )
    count = int(os.environ["J3252_FIXTURE_ROUNDS"])
    assert count in {1, 2, 3}
    with bootstrap_execution(args, process_start=True):
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        for _ in range(count):
            state = grow(args)
            run_dataset(args, generation=state.generation)
            accepted = run_pipeline_training_stage(args, generation=state.generation)
            assert (
                run_pipeline_training_stage(args, generation=state.generation)
                == accepted
            )
        atomic_json(root.parent / (root.name + "-work-done.json"), {"rounds": count})
        while True:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
