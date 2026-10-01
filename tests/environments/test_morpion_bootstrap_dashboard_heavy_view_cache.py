"""Regression tests for persistent Record/Tree dashboard caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.training_export import TrainingTreeSnapshot, save_training_tree_snapshot
from atomheart.games.morpion import MorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

import chipiron.environments.morpion.bootstrap.dashboard.history_view as history_view_module
import chipiron.environments.morpion.bootstrap.dashboard.record_view_cache as record_view_cache_module
import chipiron.environments.morpion.bootstrap.dashboard.tree_index_cache as tree_index_cache_module
import chipiron.environments.morpion.bootstrap.dashboard.tree_inspector as tree_inspector_module
from chipiron.environments.morpion.bootstrap import (
    AnemoneMorpionSearchRunner,
    MorpionBootstrapPaths,
    MorpionBootstrapRecordStatus,
    save_pipeline_dataset_status_file,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
    build_current_certified_record_board_view,
)
from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
    build_morpion_bootstrap_tree_inspector_snapshot,
)
from tests.environments.morpion_training_snapshot_helpers import (
    make_training_node_snapshot,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def _create_runtime_checkpoint(work_dir: Path) -> Path:
    """Create one small real runtime checkpoint for cache persistence tests."""
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


def _morpion_payload(move_count: int) -> dict[str, object]:
    """Build one real Morpion state-ref payload after a fixed number of moves."""
    dynamics = MorpionDynamics()
    state = morpion_initial_state()
    for _ in range(move_count):
        action = dynamics.all_legal_actions(state)[0]
        state = dynamics.step(state, action).next_state
    return cast("dict[str, object]", MorpionStateCheckpointCodec().dump_state_ref(state))


def test_tree_index_survives_process_cache_reset(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A second inspector session should reuse SQLite without checkpoint deserialization."""
    cache_dir = tmp_path / "dashboard-cache"
    work_dir = tmp_path / "run"
    monkeypatch.setenv("CHIPIRON_DASHBOARD_CACHE_DIR", str(cache_dir))
    _create_runtime_checkpoint(work_dir)

    tree_inspector_module._INDEXED_CHECKPOINT_TREE_CACHE.clear()
    tree_inspector_module._build_selected_node_snapshot_parts.cache_clear()
    first = build_morpion_bootstrap_tree_inspector_snapshot(work_dir)
    assert first.node_summary is not None
    assert list(cache_dir.rglob("*.sqlite3"))

    tree_inspector_module._INDEXED_CHECKPOINT_TREE_CACHE.clear()
    tree_inspector_module._build_selected_node_snapshot_parts.cache_clear()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("persistent Tree cache unexpectedly reread the full checkpoint")

    monkeypatch.setattr(
        tree_index_cache_module,
        "read_inspection_checkpoint",
        forbidden,
    )
    monkeypatch.setattr(
        tree_inspector_module,
        "read_inspection_checkpoint",
        forbidden,
    )

    second = build_morpion_bootstrap_tree_inspector_snapshot(work_dir)
    assert second.node_summary == first.node_summary
    assert second.child_summaries == first.child_summaries
    assert second.state_view == first.state_view
    assert second.local_tree_view == first.local_tree_view


def test_record_snapshot_cache_survives_dashboard_restart(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Legacy runs should scan a tree once and reuse the external record cache."""
    cache_dir = tmp_path / "dashboard-cache"
    work_dir = tmp_path / "run"
    monkeypatch.setenv("CHIPIRON_DASHBOARD_CACHE_DIR", str(cache_dir))
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    snapshot = TrainingTreeSnapshot(
        nodes=(
            make_training_node_snapshot(
                node_id="certified-2",
                parent_ids=(),
                child_ids=(),
                depth=2,
                state_ref_payload=_morpion_payload(2),
                direct_value_scalar=2.0,
                backed_up_value_scalar=2.0,
                is_terminal=True,
                is_exact=True,
                over_event_label=None,
                visit_count=3,
                metadata={"source": "persistent-record-cache-test"},
            ),
        ),
        root_node_id="certified-2",
    )
    save_training_tree_snapshot(
        snapshot,
        paths.tree_snapshot_dir / "generation_000001.json",
    )

    first = build_current_certified_record_board_view(work_dir)
    assert first is not None
    assert first.total_points == 38
    assert list(cache_dir.rglob("*.json"))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("persistent Record cache unexpectedly rescanned the tree")

    monkeypatch.setattr(
        history_view_module,
        "_load_resolved_training_tree_snapshot",
        forbidden,
    )
    second = build_current_certified_record_board_view(work_dir)
    assert second == first


def test_record_uses_existing_leaderboard_state_without_tree_snapshot(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Modern dataset status + leaderboard state should make Record a tiny-file read."""
    cache_dir = tmp_path / "dashboard-cache"
    work_dir = tmp_path / "run"
    leaderboard_path = tmp_path / "morpion_leaderboard.jsonl"
    monkeypatch.setenv("CHIPIRON_DASHBOARD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        record_view_cache_module,
        "_default_leaderboard_path",
        lambda: leaderboard_path,
    )

    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    status = MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=2,
        current_best_total_points=38,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )
    save_pipeline_dataset_status_file(
        generation=1,
        dataset_status="done",
        updated_at_utc="2026-10-01T09:00:00Z",
        metadata={"source": "leaderboard-fast-path-test"},
        record_status=status,
        path=paths.pipeline_dataset_status_path_for_generation(1),
    )
    leaderboard_path.write_text(
        json.dumps(
            {
                "variant": "5T",
                "total_points": 38,
                "moves_since_start": 2,
                "is_terminal": True,
                "is_exact": True,
                "source": "certified_terminal_leaf",
                "state_fingerprint": "sha256:test",
                "state_ref_payload": checkpoint_payload_to_jsonable(
                    _morpion_payload(2)
                ),
                "run_work_dir": str(work_dir.resolve()),
                "generation": 1,
                "cycle_index": 1,
                "timestamp_utc": "2026-10-01T09:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("leaderboard Record fast path unexpectedly loaded a tree snapshot")

    monkeypatch.setattr(
        history_view_module,
        "_load_resolved_training_tree_snapshot",
        forbidden,
    )
    board = build_current_certified_record_board_view(work_dir)
    assert board is not None
    assert board.moves_since_start == 2
    assert board.total_points == 38
    assert board.source == "certified_terminal_leaf"
    assert list(cache_dir.rglob("*.json"))
