"""Resolve Dataset preparation from the same parent-independent candidate arguments."""

import os
from dataclasses import replace
from typing import TYPE_CHECKING

from .candidate_preparation import PreparationRecipe, preparation_recipe

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs


def dataset_recipe(args: MorpionBootstrapArgs, rows: Path) -> PreparationRecipe | None:
    """Resolve future Training inputs without constructing a continuation/parent."""
    if os.environ.get("CHIPIRON_SHARED_PREPARATION", "0") == "0":
        return None
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionStreamingTrainingArgs,
    )

    from .candidate_resources import _integer
    from .cycle_training import (
        morpion_training_args_from_evaluator_spec,
        restrict_evaluators_config,
    )

    evaluators = restrict_evaluators_config(
        args.resolved_evaluators_config(), args.training_evaluator_names
    )
    streams = [
        MorpionStreamingTrainingArgs(
            training_args=replace(
                morpion_training_args_from_evaluator_spec(
                    spec=spec,
                    dataset_file=rows,
                    output_dir=rows.parent / "unused",
                    shuffle=args.shuffle,
                    validation_fraction=args.validation_fraction,
                    validation_seed=args.validation_seed,
                    device=args.training_device,
                ),
                stable_validation_split=args.continual_learning is not None,
            ),
            max_rows=args.training_max_rows,
            row_chunk_size=args.training_row_chunk_size,
        )
        for spec in evaluators.evaluators.values()
    ]
    return preparation_recipe(streams, serial=_integer("WORKERS", 1) == 1)
