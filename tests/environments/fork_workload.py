"""Finite deterministic work through ordinary owned stages, not a training loop."""

import sys
from dataclasses import replace
from pathlib import Path

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.execution_profile import (
    bootstrap_execution,
)
from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow


def main() -> None:
    """Return a clean supervisor exit only after fixed accepted work completes."""
    root = Path(sys.argv[1])
    mode = (
        Path(sys.argv[3]).read_text() if sys.argv[2] == "--mode-file" else sys.argv[2]
    )
    args = replace(
        _scientific_args(
            MorpionBootstrapArgs(root),
            load_bootstrap_config(root / "bootstrap_config.json"),
        ),
        max_rows=128,
    )
    with bootstrap_execution(args, process_start=True):
        for number in range(2):
            state = grow(args)
            if mode == "node":
                return
            run_dataset(args, generation=state.generation)
            accepted = run_pipeline_training_stage(args, generation=state.generation)
            assert (
                run_pipeline_training_stage(args, generation=state.generation)
                == accepted
            )
            if number == 0 and mode == "parent":
                paths = MorpionBootstrapPaths.from_work_dir(root)
                activate_model(
                    paths,
                    prepare_model_activation_request(
                        paths, training_generation=state.generation
                    ),
                )


if __name__ == "__main__":
    main()
