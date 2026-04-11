"""Tests for Morpion bootstrap record-status semantics."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from atomheart.games.morpion import MorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapRunState,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    bootstrap_event_from_dict,
    bootstrap_event_to_dict,
    extract_morpion_record_status_from_training_tree_snapshot,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)


def _make_morpion_payload(move_count: int) -> dict[str, object]:
    """Build one real Morpion checkpoint payload after ``move_count`` moves."""
    dynamics = MorpionDynamics()
    state = morpion_initial_state()
    for _ in range(move_count):
        action = dynamics.all_legal_actions(state)[0]
        state = dynamics.step(state, action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(state))


def _make_training_node(
    *,
    node_id: str,
    move_count: int,
    is_exact: bool = True,
    is_terminal: bool = False,
) -> TrainingNodeSnapshot:
    """Build one training-node snapshot with a known Morpion move count."""
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=(),
        child_ids=(),
        depth=move_count,
        state_ref_payload=_make_morpion_payload(move_count),
        direct_value_scalar=float(move_count),
        backed_up_value_scalar=float(move_count),
        is_terminal=is_terminal,
        is_exact=is_exact,
        over_event_label=None,
        visit_count=move_count + 1,
        metadata={"source": "record-status-test"},
    )


def test_record_status_round_trips_through_event_and_run_state(tmp_path: Path) -> None:
    """Structured record status should survive event and run-state serialization."""
    record_status = MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=18,
        current_best_total_points=54,
        current_best_is_exact=True,
        current_best_source="snapshot_exact_node",
    )
    event = MorpionBootstrapEvent(
        event_id="cycle_000005",
        cycle_index=5,
        generation=3,
        timestamp_utc="2026-04-11T08:15:00Z",
        tree=MorpionBootstrapTreeStatus(num_nodes=42),
        dataset=MorpionBootstrapDatasetStatus(num_rows=12, num_samples=12),
        training=MorpionBootstrapTrainingStatus(triggered=True),
        record=record_status,
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path="tree_exports/generation_000003.json",
            rows_path="rows/generation_000003.json",
        ),
    )
    run_state = MorpionBootstrapRunState(
        generation=3,
        cycle_index=5,
        latest_tree_snapshot_path="tree_exports/generation_000003.json",
        latest_rows_path="rows/generation_000003.json",
        latest_model_bundle_paths={"default": "models/generation_000003/default"},
        active_evaluator_name="default",
        tree_size_at_last_save=42,
        last_save_unix_s=1234.5,
        latest_record_status=record_status,
    )

    assert bootstrap_event_from_dict(bootstrap_event_to_dict(event)) == event

    path = tmp_path / "run_state.json"
    save_bootstrap_run_state(run_state, path)
    assert load_bootstrap_run_state(path) == run_state


def test_snapshot_record_status_computes_moves_since_start_and_total_points() -> None:
    """Snapshot extraction should distinguish moves since start from total points."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-2",
        nodes=(_make_training_node(node_id="node-2", move_count=2),),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    status = extract_morpion_record_status_from_training_tree_snapshot(snapshot)

    assert status == MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=2,
        current_best_total_points=38,
        current_best_is_exact=True,
        current_best_source="snapshot_exact_node",
    )


def test_snapshot_record_status_prefers_larger_achievement() -> None:
    """The best record should come from the node with the largest move count."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-3",
        nodes=(
            _make_training_node(node_id="node-1", move_count=1),
            _make_training_node(node_id="node-3", move_count=3),
            _make_training_node(node_id="node-2", move_count=2),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    status = extract_morpion_record_status_from_training_tree_snapshot(snapshot)

    assert status.current_best_moves_since_start == 3
    assert status.current_best_total_points == 39


def test_snapshot_record_status_prefers_exact_node_on_tie() -> None:
    """Exact nodes should win ties when the move count is the same."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-exact",
        nodes=(
            _make_training_node(
                node_id="node-nonexact",
                move_count=4,
                is_exact=False,
            ),
            _make_training_node(
                node_id="node-exact",
                move_count=4,
                is_exact=True,
            ),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    status = extract_morpion_record_status_from_training_tree_snapshot(snapshot)

    assert status.current_best_moves_since_start == 4
    assert status.current_best_total_points == 40
    assert status.current_best_is_exact is True
    assert status.current_best_source == "snapshot_exact_node"
