"""Tests for Morpion bootstrap record-status semantics."""
# ruff: noqa: E402

from __future__ import annotations

import json
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
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapRunState,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    bootstrap_event_from_dict,
    bootstrap_event_to_dict,
    current_record_score,
    extract_morpion_record_status_from_training_tree_snapshot,
    load_bootstrap_run_state,
    persist_certified_leaderboard_candidates,
    resolve_frontier_status_for_cycle,
    resolve_record_status_for_cycle,
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
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
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
        frontier=MorpionBootstrapFrontierStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=19,
            current_best_total_points=55,
            current_best_is_exact=False,
            current_best_is_terminal=False,
            current_best_source="snapshot_nonterminal_node",
        ),
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
        latest_frontier_status=event.frontier,
    )

    assert bootstrap_event_from_dict(bootstrap_event_to_dict(event)) == event

    path = tmp_path / "run_state.json"
    save_bootstrap_run_state(run_state, path)
    assert load_bootstrap_run_state(path) == run_state


def test_exact_terminal_node_updates_certified_record() -> None:
    """Certified records should come only from exact terminal nodes."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-2",
        nodes=(
            _make_training_node(
                node_id="node-2",
                move_count=2,
                is_exact=True,
                is_terminal=True,
            ),
        ),
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
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )


def test_current_record_score_returns_moves_since_start() -> None:
    """The canonical record score must ignore total occupied points."""
    status = MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=18,
        current_best_total_points=54,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )

    assert current_record_score(status) == 18


def test_nonterminal_non_exact_snapshot_node_does_not_update_certified_record() -> None:
    """Nonterminal estimated nodes must never update the certified record."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-3",
        nodes=(
            _make_training_node(
                node_id="node-3",
                move_count=3,
                is_exact=False,
                is_terminal=False,
            ),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    previous = MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=1,
        current_best_total_points=37,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )
    status = resolve_record_status_for_cycle(
        snapshot=snapshot,
        previous_record_status=previous,
    )

    assert status == previous


def test_frontier_best_can_update_without_affecting_certified_record() -> None:
    """Frontier progress should remain visible without contaminating certification."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-frontier",
        nodes=(
            _make_training_node(
                node_id="node-frontier",
                move_count=4,
                is_exact=False,
                is_terminal=False,
            ),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    certified_status = resolve_record_status_for_cycle(
        snapshot=snapshot,
        previous_record_status=None,
    )
    frontier_status = resolve_frontier_status_for_cycle(
        snapshot=snapshot,
        previous_frontier_status=None,
    )

    assert certified_status.current_best_total_points is None
    assert frontier_status.current_best_total_points == 40
    assert frontier_status.current_best_source == "snapshot_nonterminal_node"


def test_leaderboard_deduplicates_identical_states_by_fingerprint(
    tmp_path: Path,
) -> None:
    """Certified leaderboard entries should be deduplicated by state fingerprint."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-2",
        nodes=(
            _make_training_node(
                node_id="node-2",
                move_count=2,
                is_exact=True,
                is_terminal=True,
            ),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )
    leaderboard_path = tmp_path / "leaderboard.jsonl"

    persist_certified_leaderboard_candidates(
        snapshot=snapshot,
        run_work_dir=tmp_path,
        generation=1,
        cycle_index=0,
        timestamp_utc="2026-04-14T18:26:38Z",
        leaderboard_path=leaderboard_path,
    )
    persist_certified_leaderboard_candidates(
        snapshot=snapshot,
        run_work_dir=tmp_path,
        generation=2,
        cycle_index=1,
        timestamp_utc="2026-04-14T18:26:39Z",
        leaderboard_path=leaderboard_path,
    )

    lines = leaderboard_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_leaderboard_keeps_only_top_100_per_variant(tmp_path: Path) -> None:
    """Certified leaderboard should keep only the best 100 entries per variant."""
    leaderboard_path = tmp_path / "leaderboard.jsonl"
    for move_count in range(101):
        snapshot = TrainingTreeSnapshot(
            root_node_id=f"node-{move_count}",
            nodes=(
                _make_training_node(
                    node_id=f"node-{move_count}",
                    move_count=move_count,
                    is_exact=True,
                    is_terminal=True,
                ),
            ),
            metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
        )
        persist_certified_leaderboard_candidates(
            snapshot=snapshot,
            run_work_dir=tmp_path,
            generation=move_count,
            cycle_index=move_count,
            timestamp_utc=f"2026-04-14T18:26:{move_count:02d}Z",
            leaderboard_path=leaderboard_path,
        )

    lines = leaderboard_path.read_text(encoding="utf-8").splitlines()
    loaded = [json.loads(line) for line in lines]

    assert len(loaded) == 100
    assert loaded[0]["total_points"] == 136
    assert loaded[-1]["total_points"] == 37


def test_old_run_state_without_frontier_field_still_loads_safely(
    tmp_path: Path,
) -> None:
    """Legacy run-state payloads should still load cleanly without frontier status."""
    path = tmp_path / "run_state.json"
    path.write_text(
        json.dumps(
            {
                "generation": 3,
                "cycle_index": 17,
                "tree_size_at_last_save": 42,
                "last_save_unix_s": 1234.5,
                "latest_record_status": {
                    "variant": "5T",
                    "initial_pattern": "greek_cross",
                    "initial_point_count": 36,
                    "current_best_moves_since_start": 18,
                    "current_best_total_points": 54,
                    "current_best_is_exact": True,
                    "current_best_source": "snapshot_exact_node",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_bootstrap_run_state(path)

    assert loaded.latest_frontier_status is None
    assert loaded.latest_record_status is not None
    assert loaded.latest_record_status.current_best_is_terminal is None
