"""Tiny fresh Training workspaces; no historical adoption or service workers."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_closed_helpers import setup_closed_baseline
from tests.environments.morpion_durable_v2_helpers import scientific_producer
from tests.environments.morpion_fenced_dataset_helpers import run_dataset

if TYPE_CHECKING:
    from pathlib import Path


def setup_training(root: Path) -> Any:
    """Prepare real two-row Dataset and two tiny CPU candidate models."""
    args, paths, state = setup_closed_baseline(root, initialize=False)
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_active_model,
        save_pipeline_active_model,
    )

    active = load_pipeline_active_model(paths.pipeline_active_model_path)
    save_pipeline_active_model(
        replace(
            active,
            source="external_seed",
            source_generation=1,
            local_trained_generation=None,
        ),
        paths.pipeline_active_model_path,
    )
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=scientific_producer(args),
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        fenced_dataset=True,
        durable_training=True,
    )
    training_args = replace(
        args,
        use_backed_up_value=False,
        skip_evaluator_diagnostics=True,
        training_device="cpu",
        training_row_chunk_size=1,
        validation_fraction=0.5,
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                name: MorpionEvaluatorSpec(
                    name=name,
                    model_type=kind,
                    hidden_sizes=hidden,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=0.001,
                )
                for name, kind, hidden in (
                    ("linear", "linear", None),
                    ("small_mlp", "mlp", (4,)),
                )
            }
        ),
    )
    run_dataset(training_args)
    return training_args, paths, state
