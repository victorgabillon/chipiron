"""Tests for Morpion growth operator recaps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.growth_recap import (
    collect_growth_recap,
    render_growth_recap,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    MorpionBootstrapRunState,
    save_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_growth_recap_extracts_active_evaluator_from_state(
    tmp_path: Path,
) -> None:
    """Growth recap should summarize the active model and latest growth state."""
    run_state = MorpionBootstrapRunState(
        generation=38,
        cycle_index=42,
        latest_tree_snapshot_path="tree_exports/generation_000038.json.zst",
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name="stale_from_run_state",
        tree_size_at_last_save=242_680,
        last_save_unix_s=0.0,
        latest_runtime_checkpoint_path="search_checkpoints/generation_000038.sharded",
        metadata={
            "checkpoint": {"total_s": 93.8},
            "restore": {"total_s": 1196.8},
            "tree": {
                "branch_count": 244_584,
                "growth_elapsed_s": 590.9,
                "node_count": 242_680,
            },
        },
    )
    save_bootstrap_run_state(run_state, tmp_path / "run_state.json")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=430,
            evaluator_name="mlp_41",
            model_bundle_path="models/generation_000430/mlp_41",
            updated_at_utc="2026-07-08T10:00:00Z",
            source="external_seed",
        ),
        tmp_path / "pipeline" / "active_model.json",
    )

    recap = collect_growth_recap(tmp_path, worker_max_cycles="20")

    assert recap.latest_generation == 38
    assert recap.cycle_index == 42
    assert recap.active_evaluator_name == "mlp_41"
    assert recap.active_model_generation == 430
    assert recap.active_model_source == "external_seed"
    assert (
        recap.latest_checkpoint_path == "search_checkpoints/generation_000038.sharded"
    )
    assert recap.tree_nodes == 242_680
    assert recap.branch_count == 244_584
    assert recap.worker_max_cycles == "20"


def test_growth_recap_renders_active_evaluator_line(tmp_path: Path) -> None:
    """Rendered growth recap should make the active evaluator obvious."""
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=38,
            cycle_index=42,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name="mlp_41",
            tree_size_at_last_save=242_680,
            last_save_unix_s=0.0,
            latest_runtime_checkpoint_path="search_checkpoints/generation_000038.sharded",
        ),
        tmp_path / "run_state.json",
    )

    rendered = render_growth_recap(
        collect_growth_recap(tmp_path, worker_max_cycles="20"),
        force_plain=True,
    )

    assert "ACTIVE EVALUATOR: mlp_41" in rendered
    assert "latest_generation=38 cycle=42" in rendered
    assert "worker_max_cycles=20" in rendered
