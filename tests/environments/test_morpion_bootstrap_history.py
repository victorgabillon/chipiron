"""Tests for Morpion bootstrap history persistence and loop integration."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

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

import chipiron.environments.morpion.bootstrap.bootstrap_loop as bootstrap_loop_module
from chipiron.environments.morpion.bootstrap import (
    MORPION_BOOTSTRAP_INITIAL_PATTERN,
    MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    MORPION_BOOTSTRAP_VARIANT,
    MorpionBootstrapArgs,
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryPaths,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapLatestStatus,
    MorpionBootstrapPaths,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapRunState,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    bootstrap_event_from_dict,
    bootstrap_event_to_dict,
    default_morpion_record_status,
    latest_status_from_dict,
    latest_status_to_dict,
    load_bootstrap_history,
    load_latest_bootstrap_status,
    rebuild_latest_bootstrap_status,
    run_one_bootstrap_cycle,
)
from chipiron.environments.morpion.bootstrap.history import (
    MalformedMorpionBootstrapHistoryError,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_snapshot(
    *,
    target_value: float,
    root_node_id: str,
) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for bootstrap history tests."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=True,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-history-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the Morpion bootstrap protocol."""

    def __init__(
        self,
        *,
        tree_sizes: tuple[int, ...],
        target_values: tuple[float, ...],
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._cycle_index = -1

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Ignore inputs for the fake runner."""
        _ = tree_snapshot_path, model_bundle_path, effective_runtime_config, reevaluate_tree

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        _ = max_growth_steps
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(
        self,
        output_path: str | Path,
    ) -> None:
        """Write one real training snapshot to ``output_path``."""
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        save_training_tree_snapshot(snapshot, output_path)

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


def _multi_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one representative two-evaluator bootstrap config."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
            "mlp": MorpionEvaluatorSpec(
                name="mlp",
                model_type="mlp",
                hidden_sizes=(8, 4),
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
        }
    )


def _patch_reported_losses(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_by_evaluator_name: dict[str, float],
) -> None:
    """Patch training so history tests can assert deterministic winner selection."""
    real_train = bootstrap_loop_module.train_morpion_regressor

    def _patched_train(train_args: object) -> object:
        _model, metrics = real_train(train_args)
        evaluator_name = Path(str(train_args.output_dir)).name
        metrics["final_loss"] = loss_by_evaluator_name[evaluator_name]
        return _model, metrics

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _patched_train
    )


def _make_event(generation: int = 3, cycle_index: int = 5) -> MorpionBootstrapEvent:
    """Build one representative bootstrap event for serialization tests."""
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}",
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc="2026-04-11T08:15:00Z",
        tree=MorpionBootstrapTreeStatus(
            num_nodes=42,
            num_expanded_nodes=None,
            num_simulations=None,
            root_visit_count=None,
            min_depth_present=0,
            max_depth_present=3,
            depth_node_counts={0: 1, 1: 8, 2: 21, 3: 12},
        ),
        dataset=MorpionBootstrapDatasetStatus(num_rows=12, num_samples=12),
        training=MorpionBootstrapTrainingStatus(triggered=True),
        record=MorpionBootstrapRecordStatus(
            variant=MORPION_BOOTSTRAP_VARIANT,
            initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
            initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
            current_best_moves_since_start=17,
            current_best_total_points=53,
            current_best_is_exact=True,
            current_best_is_terminal=True,
            current_best_source="certified_terminal_leaf",
        ),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path="tree_exports/generation_000003.json",
            rows_path="rows/generation_000003.json",
            model_bundle_paths={"default": "models/generation_000003/default"},
        ),
        evaluators={
            "default": MorpionEvaluatorMetrics(
                final_loss=0.25,
                num_epochs=1,
                num_samples=12,
                metadata={"phase": "train"},
            )
        },
        metadata={
            "note": "cycle",
            "game": "morpion",
            "variant": "5T",
            "initial_pattern": "greek_cross",
            "initial_point_count": 36,
            "active_evaluator_name": "default",
            "selected_evaluator_name": "default",
            "selection_policy": "lowest_final_loss",
        },
    )


def test_bootstrap_event_serialization_round_trip() -> None:
    """Bootstrap events should round-trip through dict serialization."""
    event = _make_event()

    loaded = bootstrap_event_from_dict(bootstrap_event_to_dict(event))

    assert loaded == event


def test_latest_status_serialization_round_trip(tmp_path: Path) -> None:
    """Latest status should round-trip through dict serialization."""
    status = MorpionBootstrapLatestStatus(
        work_dir=str(tmp_path),
        latest_generation=3,
        latest_cycle_index=5,
        latest_event=_make_event(),
        metadata={"ui": "ready"},
    )

    loaded = latest_status_from_dict(latest_status_to_dict(status))

    assert loaded == status


def test_history_recorder_appends_events_in_order(tmp_path: Path) -> None:
    """History recorder should append events in order without clobbering."""
    paths = MorpionBootstrapHistoryPaths(
        work_dir=tmp_path,
        history_jsonl_path=tmp_path / "history.jsonl",
        latest_status_path=tmp_path / "latest_status.json",
    )
    recorder = MorpionBootstrapHistoryRecorder(paths)
    first_event = _make_event(generation=1, cycle_index=0)
    second_event = _make_event(generation=2, cycle_index=1)

    recorder.append_event(first_event)
    recorder.append_event(second_event)

    assert load_bootstrap_history(paths.history_jsonl_path) == (
        first_event,
        second_event,
    )


def test_record_writes_richer_latest_status(tmp_path: Path) -> None:
    """record() should append history and refresh the richer latest status."""
    paths = MorpionBootstrapHistoryPaths(
        work_dir=tmp_path,
        history_jsonl_path=tmp_path / "history.jsonl",
        latest_status_path=tmp_path / "latest_status.json",
    )
    recorder = MorpionBootstrapHistoryRecorder(paths)
    event = _make_event()

    recorder.record(event)

    assert load_bootstrap_history(paths.history_jsonl_path) == (event,)
    assert load_latest_bootstrap_status(paths.latest_status_path) == (
        MorpionBootstrapLatestStatus(
            work_dir=str(tmp_path),
            latest_generation=event.generation,
            latest_cycle_index=event.cycle_index,
            latest_event=event,
        )
    )


def test_rebuild_latest_status_uses_history_as_source_of_truth(tmp_path: Path) -> None:
    """The latest-status snapshot should be rebuildable from append-only history."""
    paths = MorpionBootstrapHistoryPaths(
        work_dir=tmp_path,
        history_jsonl_path=tmp_path / "history.jsonl",
        latest_status_path=tmp_path / "latest_status.json",
    )
    recorder = MorpionBootstrapHistoryRecorder(paths)
    recorder.append_event(_make_event(generation=1, cycle_index=0))
    latest_event = _make_event(generation=2, cycle_index=1)
    recorder.append_event(latest_event)

    rebuilt = rebuild_latest_bootstrap_status(paths)

    assert rebuilt == MorpionBootstrapLatestStatus(
        work_dir=str(tmp_path),
        latest_generation=2,
        latest_cycle_index=1,
        latest_event=latest_event,
    )
    assert load_latest_bootstrap_status(paths.latest_status_path) == rebuilt


def test_malformed_history_line_fails_loudly(tmp_path: Path) -> None:
    """Malformed JSONL history should raise the explicit history error."""
    path = tmp_path / "history.jsonl"
    path.write_text('{"generation": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(MalformedMorpionBootstrapHistoryError):
        load_bootstrap_history(path)


def test_non_integral_float_is_rejected_during_event_load() -> None:
    """Integer-like fields should reject non-integral floats instead of truncating."""
    payload = bootstrap_event_to_dict(_make_event())
    payload["cycle_index"] = 1.5

    with pytest.raises(MalformedMorpionBootstrapHistoryError):
        bootstrap_event_from_dict(payload)


def test_legacy_record_current_migrates_to_structured_record_status() -> None:
    """Legacy scalar record payloads should preserve their score semantically."""
    payload = bootstrap_event_to_dict(_make_event())
    payload["record"] = {"current": 17}

    loaded = bootstrap_event_from_dict(payload)

    assert loaded.record == MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=17,
        current_best_total_points=53,
        current_best_is_exact=None,
        current_best_source="legacy_record_current_migrated",
    )


def test_missing_latest_status_loads_as_empty_status(tmp_path: Path) -> None:
    """Missing latest-status files should load as an empty default snapshot."""
    status = load_latest_bootstrap_status(tmp_path / "latest_status.json")

    assert status == MorpionBootstrapLatestStatus(
        work_dir=str(tmp_path),
        latest_generation=None,
        latest_cycle_index=None,
        latest_event=None,
    )


def test_default_record_status_uses_current_experiment_identity() -> None:
    """The default record status should expose the current Morpion experiment."""
    status = default_morpion_record_status()

    assert status == MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_source=None,
    )


def test_bootstrap_loop_writes_history_on_no_save_cycle(tmp_path: Path) -> None:
    """Even a no-save cycle should emit one structured history event."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(15,), target_values=(1.25,))
    run_state = MorpionBootstrapRunState(
        generation=2,
        cycle_index=8,
        latest_tree_snapshot_path="tree_exports/generation_000002.json",
        latest_rows_path="rows/generation_000002.json",
        latest_model_bundle_paths={"linear": "models/generation_000002/linear"},
        active_evaluator_name="linear",
        tree_size_at_last_save=10,
        last_save_unix_s=100.0,
        latest_record_status=MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=12,
            current_best_total_points=48,
            current_best_is_exact=True,
            current_best_is_terminal=True,
            current_best_source="certified_terminal_leaf",
        ),
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=110.0,
    )

    history = load_bootstrap_history(paths.history_jsonl_path)
    latest_status = load_latest_bootstrap_status(paths.latest_status_path)

    assert next_state.generation == run_state.generation
    assert next_state.cycle_index == 9
    assert next_state.latest_record_status == run_state.latest_record_status
    assert len(history) == 1
    event = history[0]
    assert event.event_id == "cycle_000009"
    assert event.cycle_index == 9
    assert event.generation == 2
    assert event.timestamp_utc == "1970-01-01T00:01:50Z"
    assert event.tree.num_nodes == 15
    assert event.tree.num_expanded_nodes is None
    assert event.tree.num_simulations is None
    assert event.tree.root_visit_count is None
    assert event.dataset.num_rows is None
    assert event.dataset.num_samples is None
    assert not event.training.triggered
    assert event.record == MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=12,
        current_best_total_points=48,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )
    assert event.artifacts.tree_snapshot_path is None
    assert event.artifacts.rows_path is None
    assert event.artifacts.model_bundle_paths == {}
    assert event.evaluators == {}
    assert event.metadata["game"] == "morpion"
    assert event.metadata["variant"] == "5T"
    assert event.metadata["initial_pattern"] == "greek_cross"
    assert event.metadata["initial_point_count"] == 36
    assert event.metadata["active_evaluator_name"] == "linear"
    assert event.metadata["bootstrap_applied_runtime_control"] == {
        "tree_branch_limit": None
    }
    assert event.metadata["bootstrap_effective_runtime"] == {"tree_branch_limit": 128}
    assert isinstance(event.metadata["bootstrap_effective_runtime_hash"], str)
    assert latest_status.latest_generation == 2
    assert latest_status.latest_cycle_index == 9
    assert latest_status.latest_event == event


def test_bootstrap_loop_writes_history_on_save_train_cycle(tmp_path: Path) -> None:
    """A save/train cycle should emit one detailed structured history event."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    run_state = MorpionBootstrapRunState(
        generation=0,
        cycle_index=-1,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=200.0,
    )

    history = load_bootstrap_history(paths.history_jsonl_path)
    latest_status = load_latest_bootstrap_status(paths.latest_status_path)

    assert len(history) == 1
    event = history[0]
    assert event.event_id == "cycle_000000"
    assert event.cycle_index == 0
    assert event.generation == 1
    assert event.timestamp_utc == "1970-01-01T00:03:20Z"
    assert event.tree.num_nodes == 10
    assert event.tree.num_expanded_nodes is None
    assert event.tree.num_simulations is None
    assert event.tree.root_visit_count is None
    assert event.dataset.num_rows == 1
    assert event.dataset.num_samples == 1
    assert event.training.triggered
    assert event.artifacts.tree_snapshot_path == "tree_exports/generation_000001.json"
    assert event.artifacts.rows_path == "rows/generation_000001.json"
    assert event.artifacts.model_bundle_paths == {
        "default": "models/generation_000001/default"
    }
    assert event.metadata["game"] == "morpion"
    assert event.metadata["variant"] == "5T"
    assert event.metadata["initial_pattern"] == "greek_cross"
    assert event.metadata["initial_point_count"] == 36
    assert event.metadata["active_evaluator_name"] == "default"
    assert event.metadata["selected_evaluator_name"] == "default"
    assert event.metadata["selection_policy"] == "lowest_final_loss"
    assert event.metadata["bootstrap_applied_runtime_control"] == {
        "tree_branch_limit": None
    }
    assert event.metadata["bootstrap_effective_runtime"] == {"tree_branch_limit": 128}
    assert isinstance(event.metadata["bootstrap_effective_runtime_hash"], str)
    assert set(event.evaluators) == {"default"}
    assert event.evaluators["default"].final_loss is not None
    assert event.evaluators["default"].num_epochs == 1
    assert event.evaluators["default"].num_samples == 1
    assert next_state.generation == 1
    assert next_state.cycle_index == 0
    assert next_state.latest_tree_snapshot_path == event.artifacts.tree_snapshot_path
    assert next_state.latest_rows_path == event.artifacts.rows_path
    assert next_state.latest_model_bundle_paths == {
        "default": "models/generation_000001/default"
    }
    assert next_state.active_evaluator_name == "default"
    assert next_state.latest_record_status == event.record
    assert event.record == MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=1,
        current_best_total_points=37,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )
    assert latest_status.latest_generation == 1
    assert latest_status.latest_cycle_index == 0
    assert latest_status.latest_event == event


def test_bootstrap_loop_records_selected_winner_on_multi_evaluator_save_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-evaluator save cycle should record all metrics and the chosen winner."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.6, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=MorpionBootstrapRunState(
            generation=0,
            cycle_index=-1,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=0,
            last_save_unix_s=None,
        ),
        now_unix_s=200.0,
    )

    event = load_bootstrap_history(paths.history_jsonl_path)[0]

    assert set(event.evaluators) == {"linear", "mlp"}
    assert event.artifacts.model_bundle_paths == {
        "linear": "models/generation_000001/linear",
        "mlp": "models/generation_000001/mlp",
    }
    assert event.metadata["game"] == "morpion"
    assert event.metadata["variant"] == "5T"
    assert event.metadata["initial_pattern"] == "greek_cross"
    assert event.metadata["initial_point_count"] == 36
    assert event.metadata["active_evaluator_name"] == "mlp"
    assert event.metadata["selected_evaluator_name"] == "mlp"
    assert event.metadata["selection_policy"] == "lowest_final_loss"
    assert event.metadata["bootstrap_applied_runtime_control"] == {
        "tree_branch_limit": None
    }
    assert event.metadata["bootstrap_effective_runtime"] == {"tree_branch_limit": 128}
    assert isinstance(event.metadata["bootstrap_effective_runtime_hash"], str)
    assert event.record.variant == event.metadata["variant"]
    assert event.record.initial_pattern == event.metadata["initial_pattern"]
    assert event.record.initial_point_count == event.metadata["initial_point_count"]
    assert next_state.active_evaluator_name == "mlp"
