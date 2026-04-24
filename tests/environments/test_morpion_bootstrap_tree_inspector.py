"""Tests for the Morpion bootstrap tree inspector helpers."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.checkpoints import AnchorCheckpointStatePayload, DeltaCheckpointStatePayload

from chipiron.environments.morpion.bootstrap import (
    AnemoneMorpionSearchRunner,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    load_morpion_search_checkpoint_payload,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
    RUNTIME_CHECKPOINT_METADATA_KEY,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.tree_inspector import (
    _display_value_scalar,
    _index_checkpoint_payload,
    build_morpion_bootstrap_tree_inspector_snapshot,
    resolve_latest_runtime_checkpoint,
)


def _create_runtime_checkpoint(work_dir: Path) -> Path:
    """Create one small real runtime checkpoint for inspector tests."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    paths.ensure_directories()
    runner = AnemoneMorpionSearchRunner()
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=3),
    )
    runner.grow(2)
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(1)
    runner.save_checkpoint(checkpoint_path)
    return checkpoint_path


def test_missing_runtime_checkpoint_returns_empty_state(tmp_path: Path) -> None:
    """Missing runtime checkpoints should produce a clean empty-state snapshot."""
    snapshot = build_morpion_bootstrap_tree_inspector_snapshot(tmp_path)

    assert snapshot.checkpoint_path is None
    assert snapshot.selected_node_id is None
    assert snapshot.node_summary is None
    assert snapshot.child_summaries == ()
    assert snapshot.status_message == "No persisted runtime checkpoint available yet."


def test_resolve_latest_runtime_checkpoint_prefers_dedicated_run_state_path(
    tmp_path: Path,
) -> None:
    """Inspector checkpoint resolution should prefer the dedicated run-state field."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    checkpoint_path = _create_runtime_checkpoint(tmp_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=0,
            last_save_unix_s=None,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint_path),
            metadata={
                RUNTIME_CHECKPOINT_METADATA_KEY: paths.relative_to_work_dir(checkpoint_path)
            },
        ),
        paths.run_state_path,
    )

    resolved = resolve_latest_runtime_checkpoint(paths)

    assert resolved.checkpoint_path == checkpoint_path
    assert resolved.checkpoint_source == "run_state_latest_runtime_checkpoint_path"


def test_resolve_latest_runtime_checkpoint_falls_back_to_newest_existing_checkpoint(
    tmp_path: Path,
) -> None:
    """Inspector should recover when metadata points to a deleted checkpoint."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    checkpoint_path = _create_runtime_checkpoint(tmp_path)
    second_checkpoint_path = paths.runtime_checkpoint_path_for_generation(2)
    checkpoint_path.replace(second_checkpoint_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=0,
            last_save_unix_s=None,
            latest_runtime_checkpoint_path="search_checkpoints/generation_000001.json",
            metadata={
                RUNTIME_CHECKPOINT_METADATA_KEY: "search_checkpoints/generation_000001.json"
            },
        ),
        paths.run_state_path,
    )

    resolved = resolve_latest_runtime_checkpoint(paths)

    assert resolved.checkpoint_path == second_checkpoint_path
    assert resolved.checkpoint_source == "runtime_checkpoint_dir"
    assert resolved.status_message is not None
    assert "falling back to the latest checkpoint file on disk" in resolved.status_message


def test_display_value_priority_prefers_backed_up_then_direct_then_none() -> None:
    """Inspector display values should prefer backed-up values over direct values."""
    assert _display_value_scalar(backed_up_value=1.5, direct_value=0.3) == 1.5
    assert _display_value_scalar(backed_up_value=None, direct_value=0.3) == 0.3
    assert _display_value_scalar(backed_up_value=None, direct_value=None) is None


def test_tree_inspector_snapshot_defaults_to_root_and_extracts_children(
    tmp_path: Path,
) -> None:
    """Inspector should default to the root node and expose child rows from a real checkpoint."""
    checkpoint_path = _create_runtime_checkpoint(tmp_path)
    payload = load_morpion_search_checkpoint_payload(checkpoint_path)
    indexed = _index_checkpoint_payload(payload)
    assert all(
        isinstance(
            node_payload.state_payload,
            AnchorCheckpointStatePayload | DeltaCheckpointStatePayload,
        )
        for node_payload in payload.tree.nodes
    )

    snapshot = build_morpion_bootstrap_tree_inspector_snapshot(tmp_path)

    assert snapshot.root_node_id == str(indexed.root_node_id)
    assert snapshot.selected_node_id == str(indexed.root_node_id)
    assert snapshot.node_summary is not None
    assert snapshot.node_summary.node_id == str(indexed.root_node_id)
    assert len(snapshot.child_summaries) > 0
    assert snapshot.state_view is not None
    assert "<svg" in snapshot.state_view.board_svg


def test_tree_inspector_snapshot_falls_back_to_root_for_stale_selection(
    tmp_path: Path,
) -> None:
    """Stale selected node ids should warn and fall back to the root node."""
    checkpoint_path = _create_runtime_checkpoint(tmp_path)
    payload = load_morpion_search_checkpoint_payload(checkpoint_path)
    indexed = _index_checkpoint_payload(payload)

    snapshot = build_morpion_bootstrap_tree_inspector_snapshot(
        tmp_path,
        selected_node_id="999999",
    )

    assert snapshot.selected_node_id == str(indexed.root_node_id)
    assert snapshot.selection_warning is not None
    assert "showing the root node instead" in snapshot.selection_warning


def test_tree_inspector_snapshot_supports_parent_child_navigation(
    tmp_path: Path,
) -> None:
    """Inspector should let callers navigate to expanded children and back to parent ids."""
    _create_runtime_checkpoint(tmp_path)
    root_snapshot = build_morpion_bootstrap_tree_inspector_snapshot(tmp_path)
    expanded_child = next(
        child_summary
        for child_summary in root_snapshot.child_summaries
        if child_summary.child_node_id is not None
    )

    child_snapshot = build_morpion_bootstrap_tree_inspector_snapshot(
        tmp_path,
        selected_node_id=expanded_child.child_node_id,
    )

    assert child_snapshot.node_summary is not None
    assert child_snapshot.node_summary.parent_ids
    assert root_snapshot.selected_node_id in child_snapshot.node_summary.parent_ids
