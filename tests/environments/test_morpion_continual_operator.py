"""Fresh preparation, declared resource limits and policy downgrade guards."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continual_operator import (
    prepare_continual_experiment,
    resource_limit_reason,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    launcher_args_from_cli,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
    policy_from_metadata,
    validate_continual_context,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.fast
def test_preparation_records_nine_candidates_and_distinct_zero_search_seed(
    tmp_path: Path,
) -> None:
    """Default continual policy keeps every historical candidate and starts at one node."""
    root = tmp_path / "workspace"
    state = prepare_continual_experiment(
        MorpionBootstrapArgs(work_dir=root, max_growth_steps_per_cycle=32),
        MorpionContinualLearningPolicy(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(root)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    policy = policy_from_metadata(config.metadata)
    assert policy.training_policy == "continue_from_previous_accepted_weights"
    assert not policy.automatic_activation
    assert set(config.evaluators.evaluators) == {
        "linear_5",
        "linear_10",
        "linear_20",
        "linear_41",
        "mlp_5",
        "mlp_10",
        "mlp_20",
        "mlp_41",
        "entity_token_transformer_small",
    }
    assert state.tree_size_at_last_save == 1
    assert config.evaluator_update_policy == "future_only"
    assert not (root / "training_publications_v1").exists()
    _, bundle = committed_evaluator(root, state)
    model, model_args, manifest = load_morpion_model_bundle(bundle)
    assert torch.equal(model(torch.randn(5, model_args.input_dim)), torch.zeros(5, 1))
    assert manifest.metadata["certainty"] == "ESTIMATE"
    validate_continual_context(paths, state, policy)
    with pytest.raises(ValueError, match="silently disabled"):
        validate_continual_context(paths, state, None)
    with pytest.raises(ValueError, match="policy changed"):
        validate_continual_context(paths, state, replace(policy, max_tree_nodes=200001))
    authority = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError, match="new workspace"):
        prepare_continual_experiment(MorpionBootstrapArgs(root), policy)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.fast
def test_cli_continual_defaults_and_explicit_policies() -> None:
    """The existing operator interface exposes actual implemented settings."""
    options = launcher_args_from_cli([
        "--scientific",
        "--scientific-operation",
        "prepare-continuous",
        "--work-dir",
        "/unused/new",
    ])
    assert options.bootstrap_args.max_growth_steps_per_cycle == 32
    assert options.bootstrap_args.search.rollout.max_extra_steps == 2
    assert (
        options.continual_policy.training_policy
        == "continue_from_previous_accepted_weights"
    )
    explicit = launcher_args_from_cli([
        "--scientific",
        "--scientific-operation",
        "prepare-continuous",
        "--work-dir",
        "/unused/new",
        "--training-policy",
        "fresh_from_scratch",
        "--optimizer-state-policy",
        "reset_each_round",
        "--automatic-activation",
    ])
    assert explicit.continual_policy.automatic_activation
    assert explicit.continual_policy.training_policy == "fresh_from_scratch"
    assert explicit.continual_policy.optimizer_state_policy == "reset_each_round"


def test_prepared_genesis_skips_learning_then_restores_for_real_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual operator preparation can enter the existing scientific workers."""
    monkeypatch.setenv("MORPION_LEADERBOARD_PATH", str(tmp_path / "leaderboard.jsonl"))
    root = tmp_path / "fresh"
    prepare_continual_experiment(
        MorpionBootstrapArgs(work_dir=root, max_growth_steps_per_cycle=2),
        MorpionContinualLearningPolicy(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(root)
    authority = paths.run_state_path.read_bytes()
    for stage in ("dataset_worker", "training_worker"):
        result = run_morpion_bootstrap_experiment(
            launcher_args_from_cli([
                "--scientific",
                "--work-dir",
                str(root),
                "--pipeline-stage",
                stage,
            ])
        )
        assert result.generation is None
    assert paths.run_state_path.read_bytes() == authority
    run_morpion_bootstrap_experiment(
        launcher_args_from_cli([
            "--scientific",
            "--work-dir",
            str(root),
            "--pipeline-stage",
            "growth",
            "--max-cycles",
            "1",
        ])
    )
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.generation == 2
    assert state.tree_size_at_last_save > 1


@pytest.mark.fast
@pytest.mark.parametrize(
    ("metric", "value", "reason"),
    [
        ("available_ram_mb", 0, "minimum_available_ram"),
        ("free_disk_mb", 0, "minimum_free_disk"),
        ("group_rss_mb", 10000, "maximum_process_group_rss"),
        ("workspace_mb", 40000, "maximum_workspace_disk"),
        ("scratch_mb", 5000, "maximum_scratch_disk"),
        ("longest_worker_seconds", 8000, "maximum_worker_runtime"),
    ],
)
def test_resource_guards_report_the_actual_limit(
    metric: str, value: int, reason: str
) -> None:
    """Each independent declared guard has a distinct durable termination reason."""
    policy = MorpionContinualLearningPolicy()
    sample = {
        "available_ram_mb": 20000,
        "free_disk_mb": 50000,
        "group_rss_mb": 1000,
        "workspace_mb": 100,
        "scratch_mb": 100,
        "longest_worker_seconds": 10,
    }
    assert resource_limit_reason(policy, sample, 1) is None
    assert resource_limit_reason(policy, {**sample, metric: value}, 1) == reason
    assert (
        resource_limit_reason(policy, sample, policy.max_elapsed_seconds)
        == "maximum_elapsed_time"
    )
    with pytest.raises(ValueError, match="nonfinite"):
        resource_limit_reason(policy, {**sample, metric: float("nan")}, 1)
