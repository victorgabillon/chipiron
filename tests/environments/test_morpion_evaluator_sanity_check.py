"""Tests for standalone Morpion evaluator sanity checks."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pytest import MonkeyPatch

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

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    save_training_tree_snapshot,
)
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

import chipiron.environments.morpion.bootstrap.evaluator_sanity_check as sanity_module
from chipiron.environments.morpion.bootstrap.bootstrap_loop import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.evaluator_sanity_check import (
    MorpionEvaluatorSanityArgs,
    build_backup_target_diagnostics,
    build_sanity_dataset_rows,
    run_evaluator_sanity_check,
    terminal_path_nodes,
    top_terminal_path_nodes,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    load_morpion_supervised_rows,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_node(
    node_id: str,
    *,
    parent_ids: tuple[str, ...] = (),
    child_ids: tuple[str, ...] = (),
    depth: int = 0,
    backed_up_value_scalar: float | None = 1.0,
    direct_value_scalar: float | None = 0.5,
    is_terminal: bool = False,
    is_exact: bool = False,
) -> TrainingNodeSnapshot:
    """Build one valid training node for sanity-check tests."""
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=parent_ids,
        child_ids=child_ids,
        depth=depth,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=direct_value_scalar,
        backed_up_value_scalar=backed_up_value_scalar,
        is_terminal=is_terminal,
        is_exact=is_exact,
        over_event_label=None,
        visit_count=depth + 1,
        metadata={"source": "sanity-test"},
    )


def _branching_snapshot() -> TrainingTreeSnapshot:
    """Build a small tree with shared ancestors and terminal leaves."""
    root = _make_node("root", child_ids=("a", "b"), depth=0)
    node_a = _make_node("a", parent_ids=("root",), child_ids=("a1", "a2"), depth=1)
    node_b = _make_node("b", parent_ids=("root",), child_ids=("b1",), depth=1)
    node_a1 = _make_node(
        "a1",
        parent_ids=("a",),
        depth=2,
        is_terminal=True,
        backed_up_value_scalar=3.0,
    )
    node_a2 = _make_node(
        "a2",
        parent_ids=("a",),
        depth=3,
        is_exact=True,
        backed_up_value_scalar=None,
        direct_value_scalar=4.0,
    )
    node_b1 = _make_node(
        "b1",
        parent_ids=("b",),
        depth=2,
        is_terminal=True,
        backed_up_value_scalar=2.0,
    )
    return TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(root, node_a, node_b, node_a1, node_a2, node_b1),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


def test_terminal_path_extraction_follows_parent_ids_to_root() -> None:
    """The single-path dataset should walk ancestors from terminal node to root."""
    snapshot = _branching_snapshot()

    selected = terminal_path_nodes(snapshot)
    rows = build_sanity_dataset_rows(
        snapshot=snapshot,
        dataset_mode="terminal_path",
    )

    assert tuple(node.node_id for node in selected) == ("root", "a", "a2")
    assert tuple(row.node_id for row in rows.rows) == ("root", "a", "a2")
    assert rows.rows[-1].target_value == 4.0
    assert rows.rows[-1].metadata["target_source"] == "direct_value_scalar"


def test_path_extraction_can_prefer_direct_values() -> None:
    """Path datasets should honor the direct-value target option."""
    rows = build_sanity_dataset_rows(
        snapshot=_branching_snapshot(),
        dataset_mode="terminal_path",
        use_backed_up_value=False,
    )

    assert rows.rows[0].node_id == "root"
    assert rows.rows[0].target_value == 0.5
    assert rows.rows[0].metadata["target_source"] == "direct_value_scalar"


def test_top_terminal_paths_deduplicates_shared_ancestors() -> None:
    """Top-terminal path extraction should not duplicate common parents."""
    snapshot = _branching_snapshot()

    selected = top_terminal_path_nodes(snapshot, max_terminal_nodes=2)

    assert tuple(node.node_id for node in selected) == ("root", "a", "a2", "a1")


def test_backup_target_diagnostics_summarize_direct_vs_backed_up() -> None:
    """Target diagnostics should quantify backed-up/direct disagreement."""
    snapshot = _branching_snapshot()
    rows = build_sanity_dataset_rows(
        snapshot=snapshot,
        dataset_mode="top_terminal_paths",
    )

    diagnostics = build_backup_target_diagnostics(
        snapshot=snapshot,
        rows=rows,
        created_at="2026-04-26T10:00:00Z",
    )

    assert diagnostics["dataset_size"] == 4
    assert diagnostics["comparable_direct_and_backed_up_count"] == 3
    assert diagnostics["summary"]["mse_backed_up_vs_direct"] == 2.25
    assert diagnostics["backed_up_row_status"]["frontier_estimate_rows"] == 2
    assert diagnostics["backed_up_row_status"]["exact_or_terminal_rows"] == 1
    worst = diagnostics["top_worst_deltas"][0]
    assert worst["node_id"] == "a1"
    assert worst["delta"] == 2.5


def test_bootstrap_like_mode_delegates_to_existing_extractor(
    monkeypatch: MonkeyPatch,
) -> None:
    """The bootstrap-like mode should call the existing snapshot converter."""
    snapshot = _branching_snapshot()
    expected_rows = MorpionSupervisedRows(
        rows=(
            MorpionSupervisedRow(
                node_id="delegated",
                state_ref_payload=_make_morpion_payload(),
                target_value=1.0,
                is_terminal=True,
                is_exact=True,
                depth=1,
            ),
        ),
        metadata={"source": "delegated"},
    )
    calls: list[dict[str, object]] = []

    def _fake_converter(
        snapshot_arg: TrainingTreeSnapshot,
        **kwargs: object,
    ) -> MorpionSupervisedRows:
        calls.append({"snapshot": snapshot_arg, **kwargs})
        return expected_rows

    monkeypatch.setattr(
        sanity_module,
        "training_tree_snapshot_to_morpion_supervised_rows",
        _fake_converter,
    )

    rows = build_sanity_dataset_rows(
        snapshot=snapshot,
        dataset_mode="bootstrap_like",
        max_rows=3,
        min_depth=2,
        use_backed_up_value=False,
    )

    assert rows == expected_rows
    assert calls == [
        {
            "snapshot": snapshot,
            "require_exact_or_terminal": False,
            "min_depth": 2,
            "min_visit_count": None,
            "max_rows": 3,
            "use_backed_up_value": False,
            "metadata": {"sanity_dataset_mode": "bootstrap_like"},
        }
    ]


def test_sanity_check_writes_rows_diagnostics_and_summary(tmp_path: Path) -> None:
    """A tiny real snapshot should produce the expected sanity artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_training_tree_snapshot(_branching_snapshot(), paths.tree_snapshot_path_for_generation(1))

    summary = run_evaluator_sanity_check(
        MorpionEvaluatorSanityArgs(
            work_dir=tmp_path,
            generation=1,
            dataset_mode="terminal_path",
            evaluator_name="linear_5",
            run_name="test_run",
            num_epochs=1,
            batch_size=2,
            shuffle=False,
        )
    )

    output_dir = tmp_path / "evaluator_sanity" / "test_run"
    rows_path = output_dir / "rows.json"
    diagnostics_path = output_dir / "diagnostics" / "linear_5.json"
    target_diagnostics_path = output_dir / "target_diagnostics.json"
    summary_path = output_dir / "summary.json"

    assert rows_path.is_file()
    assert diagnostics_path.is_file()
    assert target_diagnostics_path.is_file()
    assert summary_path.is_file()
    assert (output_dir / "models" / "linear_5").is_dir()

    rows = load_morpion_supervised_rows(rows_path)
    diagnostics_data = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    target_diagnostics_data = json.loads(
        target_diagnostics_path.read_text(encoding="utf-8")
    )
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert tuple(row.node_id for row in rows.rows) == ("root", "a", "a2")
    assert diagnostics_data["evaluator_name"] == "linear_5"
    assert diagnostics_data["dataset_size"] == 3
    assert target_diagnostics_data["dataset_size"] == 3
    assert summary["num_rows"] == 3
    assert summary["run_name"] == "test_run"
    assert summary_data["target_diagnostics_path"] == str(target_diagnostics_path)
    assert summary_data["evaluators"]["linear_5"]["num_samples"] == 3
