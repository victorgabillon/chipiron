"""Fresh real neural/checkpoint/Dataset/Training fixtures for explicit activation."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_v2_helpers import scientific_args
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_anemone_runner import _make_model_bundle


def setup_activation(
    root: Path,
    *,
    count: int = 4,
    layout: str = "sharded",
    seed_score: float | None = None,
    branch_limit: int = 100000,
) -> Any:
    """No historical inputs or fake checkpoints; train a tiny accepted candidate."""
    args = replace(
        scientific_args(root),
        runtime_checkpoint_format=layout,
        tree_branch_limit=branch_limit,
        max_growth_steps_per_cycle=1,
        save_after_seconds=0.0,
        dataset_confidence_policy="selected_value_v2",
        require_exact_or_terminal=False,
        use_backed_up_value=False,
        skip_evaluator_diagnostics=True,
        training_device="cpu",
        training_row_chunk_size=16,
        max_rows=8,
        validation_fraction=0.5,
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                "linear": MorpionEvaluatorSpec(
                    name="linear",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=4,
                    learning_rate=0.001,
                ),
            }
        ),
    )
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    bundle = _make_model_bundle(root / "seed")
    if seed_score is not None:
        import torch

        from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
            load_morpion_model_bundle,
            save_morpion_model_bundle,
        )

        model, model_args, _ = load_morpion_model_bundle(bundle)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            list(model.parameters())[-1].fill_(seed_score)
        save_morpion_model_bundle(model, bundle, model_args=model_args)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="seed",
            model_bundle_path="seed",
            updated_at_utc="now",
            source="external_seed",
            source_generation=1,
        ),
        paths.pipeline_active_model_path,
    )
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            random_seed=4,
            runtime_checkpoint_format=layout,
        )
    )
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )

    runner.load_or_create(
        None, bundle, MorpionBootstrapEffectiveRuntimeConfig(branch_limit)
    )
    while runner.current_tree_size() < count:
        runner.grow(1)
    checkpoint = root / (
        "source-checkpoint" if layout == "sharded" else "source.json.zst"
    )
    runner.save_checkpoint(checkpoint)
    producer = {
        "revision": "atomic-activation-synthetic",
        "objective": "maximum_total_moves",
        "evaluator": artifact_digest(bundle),
        "config_sha256": bootstrap_config_sha256(config),
    }
    science = runner.export_closed_scientific_snapshot(
        root / "source-science", generation=1, producer=producer
    )
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        last_save_unix_s=1e9,
        tree_size_at_last_save=runner.current_tree_size(),
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(science),
        metadata={
            "bootstrap_config_hash": bootstrap_config_sha256(config),
            "bootstrap_effective_runtime": {
                "tree_branch_limit": branch_limit,
                "reevaluation_blend_alpha": 1.0,
            },
        },
    )
    runner.discard_uncommitted_runtime()
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=producer,
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        growth_scoring=True,
        fenced_dataset=True,
        durable_training=True,
        model_activation=True,
    )
    run_dataset(args)
    run_pipeline_training_stage(args, generation=1)
    return args, paths, state
