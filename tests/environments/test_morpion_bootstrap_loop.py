"""Tests for the restartable Morpion bootstrap loop."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch

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
from torch.utils.data import DataLoader

import chipiron.environments.morpion.bootstrap.bootstrap_loop as bootstrap_loop_module
from chipiron.environments.morpion.bootstrap import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    MORPION_BOOTSTRAP_INITIAL_PATTERN,
    MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    MORPION_BOOTSTRAP_VARIANT,
    EmptyMorpionEvaluatorsConfigError,
    MalformedMorpionBootstrapRunStateError,
    MissingActiveMorpionEvaluatorError,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    NoSelectableMorpionEvaluatorError,
    UnknownActiveMorpionEvaluatorError,
    initialize_bootstrap_run_state,
    load_bootstrap_history,
    load_morpion_evaluator_from_model_bundle,
    load_latest_bootstrap_status,
    load_bootstrap_run_state,
    run_morpion_bootstrap_loop,
    run_one_bootstrap_cycle,
    save_bootstrap_run_state,
    select_active_evaluator_name,
    should_save_progress,
    canonical_morpion_evaluator_family_config,
)
from chipiron.environments.morpion.learning import load_morpion_supervised_rows
from chipiron.environments.morpion.learning import MorpionSupervisedRows
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_CANONICAL_FEATURE_NAMES,
    load_morpion_model_bundle,
)


def _feature_subset(width: int) -> tuple[str, tuple[str, ...]]:
    """Return one deterministic explicit subset selection for loop tests."""
    return (
        f"handcrafted_{width}_custom",
        MORPION_CANONICAL_FEATURE_NAMES[:width],
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
    """Build one minimal valid training snapshot for the bootstrap loop."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=False,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-test"},
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
        self.load_calls: list[tuple[str | None, str | None]] = []
        self.grow_calls: list[int] = []
        self.export_calls: list[str] = []

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
    ) -> None:
        """Record the latest tree/model inputs used to initialize the runner."""
        _ = effective_runtime_config
        self.load_calls.append(
            (
                None if tree_snapshot_path is None else str(tree_snapshot_path),
                None if model_bundle_path is None else str(model_bundle_path),
            )
        )

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        self.grow_calls.append(max_growth_steps)
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(
        self,
        output_path: str | Path,
    ) -> None:
        """Write one real training snapshot to ``output_path``."""
        self.export_calls.append(str(output_path))
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
    """Patch training so evaluator selection is deterministic while bundles still exist."""
    real_train = bootstrap_loop_module.train_morpion_regressor

    def _patched_train(train_args: object) -> object:
        _model, metrics = real_train(train_args)
        evaluator_name = Path(str(getattr(train_args, "output_dir"))).name
        metrics["final_loss"] = loss_by_evaluator_name[evaluator_name]
        return _model, metrics

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _patched_train
    )


def _empty_rows_bundle(*, generation: int = 1) -> MorpionSupervisedRows:
    """Return one explicit empty rows bundle for empty-dataset loop tests."""
    return MorpionSupervisedRows(
        rows=(),
        metadata={
            "bootstrap_generation": generation,
            "num_rows": 0,
        },
    )


def test_should_save_progress_helper() -> None:
    """The save trigger should fire on first save, growth, or elapsed time."""
    assert should_save_progress(
        current_tree_size=1,
        tree_size_at_last_save=0,
        now_unix_s=100.0,
        last_save_unix_s=None,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert should_save_progress(
        current_tree_size=20,
        tree_size_at_last_save=10,
        now_unix_s=120.0,
        last_save_unix_s=100.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert should_save_progress(
        current_tree_size=11,
        tree_size_at_last_save=10,
        now_unix_s=3700.0,
        last_save_unix_s=0.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert not should_save_progress(
        current_tree_size=15,
        tree_size_at_last_save=10,
        now_unix_s=100.0,
        last_save_unix_s=50.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )


def test_previous_effective_runtime_config_falls_back_to_persisted_baseline() -> None:
    """Legacy run metadata with only a runtime checkpoint should fall back cleanly."""
    config = bootstrap_loop_module.bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=Path("/tmp/legacy-runtime"), tree_branch_limit=96)
    )

    previous_config = bootstrap_loop_module._previous_effective_runtime_config(
        {bootstrap_loop_module.RUNTIME_CHECKPOINT_METADATA_KEY: "search_checkpoints/generation_000001.json"},
        resolved_bootstrap_config=config,
    )

    assert previous_config == bootstrap_loop_module.MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=96,
    )


def test_run_state_round_trip(tmp_path: Path) -> None:
    """Persisted bootstrap run state should round-trip cleanly."""
    state = MorpionBootstrapRunState(
        generation=3,
        cycle_index=17,
        latest_tree_snapshot_path="tree_exports/generation_000003.json",
        latest_rows_path="rows/generation_000003.json",
        latest_model_bundle_paths={
            "linear": "models/generation_000003/linear",
            "mlp": "models/generation_000003/mlp",
        },
        active_evaluator_name="mlp",
        tree_size_at_last_save=42,
        last_save_unix_s=1234.5,
        latest_record_status=bootstrap_loop_module.MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=19,
            current_best_total_points=55,
            current_best_is_exact=False,
            current_best_source="snapshot_nonterminal_node",
        ),
        metadata={"note": "checkpoint"},
    )
    path = tmp_path / "run_state.json"

    save_bootstrap_run_state(state, path)
    loaded = load_bootstrap_run_state(path)

    assert loaded == state


def test_run_state_loads_legacy_single_model_path(tmp_path: Path) -> None:
    """Older single-model run states should migrate to the keyed mapping."""
    path = tmp_path / "run_state.json"
    path.write_text(
        json.dumps(
            {
                "generation": 2,
                "cycle_index": 8,
                "latest_tree_snapshot_path": "tree_exports/generation_000002.json",
                "latest_rows_path": "rows/generation_000002.json",
                "latest_model_bundle_path": "models/generation_000002",
                "tree_size_at_last_save": 21,
                "last_save_unix_s": 123.0,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_bootstrap_run_state(path)

    assert loaded.latest_model_bundle_paths == {"default": "models/generation_000002"}
    assert loaded.active_evaluator_name == "default"


def test_malformed_run_state_load_fails_loudly(tmp_path: Path) -> None:
    """Malformed persisted run-state payloads should raise clearly."""
    path = tmp_path / "run_state.json"
    path.write_text('{"generation": "oops", "metadata": []}\n', encoding="utf-8")

    with pytest.raises(MalformedMorpionBootstrapRunStateError):
        load_bootstrap_run_state(path)


def test_empty_evaluators_config_raises_explicitly() -> None:
    """An empty evaluator config should fail with an explicit domain error."""
    with pytest.raises(EmptyMorpionEvaluatorsConfigError):
        MorpionEvaluatorsConfig(evaluators={})


def test_select_active_evaluator_name_uses_lowest_loss() -> None:
    """The active evaluator should be selected by the smallest reported final loss."""
    selected = select_active_evaluator_name(
        {
            "linear": bootstrap_loop_module.MorpionEvaluatorMetrics(
                final_loss=0.5,
                num_epochs=1,
                num_samples=1,
            ),
            "mlp": bootstrap_loop_module.MorpionEvaluatorMetrics(
                final_loss=0.1,
                num_epochs=1,
                num_samples=1,
            ),
        }
    )

    assert selected == "mlp"


def test_select_active_evaluator_name_rejects_missing_losses() -> None:
    """Selection should fail loudly if no evaluator reports a usable loss."""
    with pytest.raises(NoSelectableMorpionEvaluatorError):
        select_active_evaluator_name(
            {
                "linear": bootstrap_loop_module.MorpionEvaluatorMetrics(
                    final_loss=None,
                    num_epochs=1,
                    num_samples=1,
                )
            }
        )


def test_select_active_evaluator_name_rejects_all_nonfinite_losses() -> None:
    """Selection should fail if every reported loss is missing or non-finite."""
    with pytest.raises(NoSelectableMorpionEvaluatorError):
        select_active_evaluator_name(
            {
                "linear": bootstrap_loop_module.MorpionEvaluatorMetrics(
                    final_loss=float("nan"),
                    num_epochs=1,
                    num_samples=1,
                ),
                "mlp": bootstrap_loop_module.MorpionEvaluatorMetrics(
                    final_loss=float("inf"),
                    num_epochs=1,
                    num_samples=1,
                ),
                "default": bootstrap_loop_module.MorpionEvaluatorMetrics(
                    final_loss=None,
                    num_epochs=1,
                    num_samples=1,
                ),
            }
        )


def test_run_one_cycle_without_save_does_not_train(tmp_path: Path) -> None:
    """A cycle below both save thresholds should skip export and training."""
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
        cycle_index=11,
        latest_tree_snapshot_path="tree_exports/generation_000002.json",
        latest_rows_path="rows/generation_000002.json",
        latest_model_bundle_paths={"linear": "models/generation_000002/linear"},
        active_evaluator_name="linear",
        tree_size_at_last_save=10,
        last_save_unix_s=100.0,
        latest_record_status=bootstrap_loop_module.MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=12,
            current_best_total_points=48,
            current_best_is_exact=True,
            current_best_source="snapshot_exact_node",
        ),
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=110.0,
    )

    assert next_state.generation == 2
    assert next_state.cycle_index == 12
    assert next_state.latest_tree_snapshot_path == "tree_exports/generation_000002.json"
    assert next_state.latest_rows_path == "rows/generation_000002.json"
    assert next_state.latest_model_bundle_paths == {
        "linear": "models/generation_000002/linear"
    }
    assert next_state.active_evaluator_name == "linear"
    assert next_state.latest_record_status == bootstrap_loop_module.MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=12,
        current_best_total_points=48,
        current_best_is_exact=True,
        current_best_source="snapshot_exact_node",
    )
    assert runner.export_calls == []
    assert (
        not paths.tree_snapshot_dir.exists()
        or list(paths.tree_snapshot_dir.iterdir()) == []
    )
    assert not paths.rows_dir.exists() or list(paths.rows_dir.iterdir()) == []
    assert not paths.model_dir.exists() or list(paths.model_dir.iterdir()) == []


def test_run_one_cycle_with_save_updates_artifacts(tmp_path: Path) -> None:
    """A cycle crossing the save threshold should export rows and train a model."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    run_state = initialize_bootstrap_run_state()

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=200.0,
    )

    assert next_state.generation == 1
    assert next_state.cycle_index == 0
    assert next_state.tree_size_at_last_save == 10
    assert next_state.last_save_unix_s == 200.0
    assert next_state.latest_tree_snapshot_path == "tree_exports/generation_000001.json"
    assert next_state.latest_rows_path == "rows/generation_000001.json"
    assert next_state.latest_model_bundle_paths == {
        "default": "models/generation_000001/default"
    }
    assert next_state.active_evaluator_name == "default"
    assert next_state.latest_record_status == bootstrap_loop_module.MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=1,
        current_best_total_points=37,
        current_best_is_exact=True,
        current_best_source="snapshot_exact_node",
    )
    assert paths.resolve_work_dir_path(next_state.latest_tree_snapshot_path).is_file()
    assert paths.resolve_work_dir_path(next_state.latest_rows_path).is_file()
    assert paths.resolve_work_dir_path(
        next_state.latest_model_bundle_paths["default"]
    ).is_dir()
    event = load_bootstrap_history(paths.history_jsonl_path)[0]
    assert event.metadata["game"] == "morpion"
    assert event.metadata["variant"] == "5T"
    assert event.metadata["initial_pattern"] == "greek_cross"
    assert event.metadata["initial_point_count"] == 36
    assert event.metadata["active_evaluator_name"] == "default"
    assert event.metadata["selected_evaluator_name"] == "default"
    assert event.metadata["selection_policy"] == "lowest_final_loss"
    assert event.record == next_state.latest_record_status


def test_run_one_cycle_with_multiple_evaluators_selects_lowest_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save cycle should train all configured evaluators into separate directories."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.4, "mlp": 0.1},
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
        run_state=initialize_bootstrap_run_state(),
        now_unix_s=200.0,
    )
    history = load_bootstrap_history(paths.history_jsonl_path)

    assert next_state.latest_model_bundle_paths == {
        "linear": "models/generation_000001/linear",
        "mlp": "models/generation_000001/mlp",
    }
    assert next_state.active_evaluator_name == "mlp"
    assert next_state.latest_record_status == bootstrap_loop_module.MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=1,
        current_best_total_points=37,
        current_best_is_exact=True,
        current_best_source="snapshot_exact_node",
    )
    assert paths.resolve_work_dir_path("models/generation_000001/linear").is_dir()
    assert paths.resolve_work_dir_path("models/generation_000001/mlp").is_dir()
    assert len(history) == 1
    event = history[0]
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
    assert event.record.variant == MORPION_BOOTSTRAP_VARIANT
    assert event.record.initial_pattern == MORPION_BOOTSTRAP_INITIAL_PATTERN
    assert event.record.initial_point_count == MORPION_BOOTSTRAP_INITIAL_POINT_COUNT


def test_loop_resumes_from_saved_run_state(tmp_path: Path) -> None:
    """The full loop should resume from persisted state instead of starting over."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=2.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 20),
        target_values=(1.25, -0.5),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    first_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    second_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert (tmp_path / "run_state.json").is_file()
    assert first_state.generation == 1
    assert second_state.generation == 2
    assert first_state.cycle_index == 0
    assert second_state.cycle_index == 1
    assert runner.load_calls[0] == (None, None)
    assert runner.load_calls[1] == (
        str(paths.resolve_work_dir_path(first_state.latest_tree_snapshot_path)),
        str(
            paths.resolve_work_dir_path(
                first_state.latest_model_bundle_paths["default"]
            )
        ),
    )


def test_loop_resumes_with_selected_winning_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart should use the persisted winning evaluator bundle for bootstrap."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.7, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=2.0,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 20),
        target_values=(1.25, -0.5),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    first_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert runner.load_calls[1] == (
        str(paths.resolve_work_dir_path(first_state.latest_tree_snapshot_path)),
        str(paths.resolve_work_dir_path(first_state.latest_model_bundle_paths["mlp"])),
    )


def test_resume_fails_for_unknown_active_evaluator(tmp_path: Path) -> None:
    """Bootstrap should fail loudly when the persisted active evaluator is unknown."""
    args = MorpionBootstrapArgs(work_dir=tmp_path, max_growth_steps_per_cycle=5)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(15,), target_values=(1.25,))

    with pytest.raises(UnknownActiveMorpionEvaluatorError):
        run_one_bootstrap_cycle(
            args=args,
            paths=paths,
            runner=runner,
            run_state=MorpionBootstrapRunState(
                generation=2,
                cycle_index=11,
                latest_tree_snapshot_path="tree_exports/generation_000002.json",
                latest_rows_path="rows/generation_000002.json",
                latest_model_bundle_paths={"linear": "models/generation_000002/linear"},
                active_evaluator_name="mlp",
                tree_size_at_last_save=10,
                last_save_unix_s=100.0,
            ),
            now_unix_s=110.0,
        )


def test_resume_fails_for_missing_active_evaluator_with_multiple_bundles(
    tmp_path: Path,
) -> None:
    """Bootstrap should fail loudly when multi-bundle state has no active evaluator."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(15,), target_values=(1.25,))

    with pytest.raises(MissingActiveMorpionEvaluatorError):
        run_one_bootstrap_cycle(
            args=args,
            paths=paths,
            runner=runner,
            run_state=MorpionBootstrapRunState(
                generation=2,
                cycle_index=11,
                latest_tree_snapshot_path="tree_exports/generation_000002.json",
                latest_rows_path="rows/generation_000002.json",
                latest_model_bundle_paths={
                    "linear": "models/generation_000002/linear",
                    "mlp": "models/generation_000002/mlp",
                },
                active_evaluator_name=None,
                tree_size_at_last_save=10,
                last_save_unix_s=100.0,
            ),
            now_unix_s=110.0,
        )


def test_resume_uses_single_saved_bundle_without_active_evaluator(
    tmp_path: Path,
) -> None:
    """Single-bundle persisted state should still resume without an active name."""
    args = MorpionBootstrapArgs(work_dir=tmp_path, max_growth_steps_per_cycle=5)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(15,), target_values=(1.25,))

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=MorpionBootstrapRunState(
            generation=2,
            cycle_index=11,
            latest_tree_snapshot_path="tree_exports/generation_000002.json",
            latest_rows_path="rows/generation_000002.json",
            latest_model_bundle_paths={"linear": "models/generation_000002/linear"},
            active_evaluator_name=None,
            tree_size_at_last_save=10,
            last_save_unix_s=100.0,
        ),
        now_unix_s=110.0,
    )

    assert runner.load_calls[0] == (
        str(paths.resolve_work_dir_path("tree_exports/generation_000002.json")),
        str(paths.resolve_work_dir_path("models/generation_000002/linear")),
    )
    assert next_state.active_evaluator_name == "linear"
    event = load_bootstrap_history(paths.history_jsonl_path)[0]
    assert event.metadata["game"] == "morpion"
    assert event.metadata["variant"] == "5T"
    assert event.metadata["initial_pattern"] == "greek_cross"
    assert event.metadata["initial_point_count"] == 36
    assert event.metadata["active_evaluator_name"] == "linear"
    assert event.metadata["bootstrap_applied_runtime_control"] == {
        "tree_branch_limit": None
    }
    assert event.metadata["bootstrap_effective_runtime"] == {
        "tree_branch_limit": 128
    }
    assert isinstance(event.metadata["bootstrap_effective_runtime_hash"], str)
    assert event.record == bootstrap_loop_module.MorpionBootstrapRecordStatus(
        variant="5T",
        initial_pattern="greek_cross",
        initial_point_count=36,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_source=None,
    )


def test_saved_rows_come_from_saved_tree_export_path(tmp_path: Path) -> None:
    """Saved rows should reflect the snapshot content written for each generation."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 11),
        target_values=(1.25, -0.5),
    )

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=2)
    assert final_state.generation == 2

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    first_rows = load_morpion_supervised_rows(paths.rows_path_for_generation(1))
    second_rows = load_morpion_supervised_rows(paths.rows_path_for_generation(2))

    assert first_rows.rows[0].target_value == 1.25
    assert second_rows.rows[0].target_value == -0.5


def test_empty_dataset_first_cycle_skips_training_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty first-cycle dataset should skip training cleanly without selection."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    monkeypatch.setattr(
        bootstrap_loop_module,
        "training_tree_snapshot_to_morpion_supervised_rows",
        lambda *args, **kwargs: _empty_rows_bundle(generation=1),
    )

    def _unexpected_train(train_args: object) -> object:
        del train_args
        raise AssertionError("training should not run for an empty extracted dataset")

    def _unexpected_select(**kwargs: object) -> str:
        del kwargs
        raise AssertionError("evaluator selection should not run for an empty extracted dataset")

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _unexpected_train
    )
    monkeypatch.setattr(
        bootstrap_loop_module, "_select_active_evaluator_name", _unexpected_select
    )

    state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    history = load_bootstrap_history(paths.history_jsonl_path)
    latest_status = load_latest_bootstrap_status(paths.latest_status_path)
    saved_rows = load_morpion_supervised_rows(paths.rows_path_for_generation(1))

    assert state.generation == 1
    assert state.cycle_index == 0
    assert state.active_evaluator_name is None
    assert state.latest_model_bundle_paths is None
    assert state.latest_rows_path == "rows/generation_000001.json"
    assert state.latest_tree_snapshot_path == "tree_exports/generation_000001.json"
    assert state.metadata[bootstrap_loop_module.TRAINING_SKIPPED_REASON_METADATA_KEY] == (
        bootstrap_loop_module.EMPTY_DATASET_TRAINING_SKIPPED_REASON
    )
    assert len(history) == 1
    event = history[0]
    assert event.dataset.num_rows == 0
    assert event.dataset.num_samples == 0
    assert not event.training.triggered
    assert event.evaluators == {}
    assert event.artifacts.model_bundle_paths == {}
    assert event.metadata[bootstrap_loop_module.TRAINING_SKIPPED_REASON_METADATA_KEY] == (
        bootstrap_loop_module.EMPTY_DATASET_TRAINING_SKIPPED_REASON
    )
    assert "selected_evaluator_name" not in event.metadata
    assert latest_status.latest_event == event
    assert saved_rows.rows == ()
    assert saved_rows.metadata["num_rows"] == 0
    assert not paths.model_dir.exists() or list(paths.model_dir.rglob("*")) == []


def test_empty_dataset_resume_preserves_previous_evaluator_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty resumed cycle should preserve the previously active evaluator and bundles."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        shuffle=False,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    previous_bundle_paths = {"linear": "models/generation_000001/linear"}
    previous_state = MorpionBootstrapRunState(
        generation=1,
        cycle_index=0,
        latest_tree_snapshot_path="tree_exports/generation_000001.json",
        latest_rows_path="rows/generation_000001.json",
        latest_model_bundle_paths=previous_bundle_paths,
        active_evaluator_name="linear",
        tree_size_at_last_save=10,
        last_save_unix_s=100.0,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(20,), target_values=(0.5,))

    monkeypatch.setattr(
        bootstrap_loop_module,
        "training_tree_snapshot_to_morpion_supervised_rows",
        lambda *args, **kwargs: _empty_rows_bundle(generation=2),
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=previous_state,
        now_unix_s=200.0,
    )
    history = load_bootstrap_history(paths.history_jsonl_path)

    assert next_state.generation == 2
    assert next_state.cycle_index == 1
    assert next_state.active_evaluator_name == "linear"
    assert next_state.latest_model_bundle_paths == previous_bundle_paths
    assert next_state.latest_model_bundle_paths is not previous_bundle_paths
    assert next_state.metadata[bootstrap_loop_module.TRAINING_SKIPPED_REASON_METADATA_KEY] == (
        bootstrap_loop_module.EMPTY_DATASET_TRAINING_SKIPPED_REASON
    )
    assert len(history) == 1
    event = history[0]
    assert event.metadata["active_evaluator_name"] == "linear"
    assert event.metadata[bootstrap_loop_module.TRAINING_SKIPPED_REASON_METADATA_KEY] == (
        bootstrap_loop_module.EMPTY_DATASET_TRAINING_SKIPPED_REASON
    )
    assert "selected_evaluator_name" not in event.metadata
    assert event.evaluators == {}
    assert event.artifacts.model_bundle_paths == {}
    assert not event.training.triggered


def test_empty_dataset_cycle_does_not_call_evaluator_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-dataset cycles should bypass evaluator winner selection entirely."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    monkeypatch.setattr(
        bootstrap_loop_module,
        "training_tree_snapshot_to_morpion_supervised_rows",
        lambda *args, **kwargs: _empty_rows_bundle(generation=1),
    )

    def _unexpected_select(**kwargs: object) -> str:
        del kwargs
        raise AssertionError("selection should not be called on an empty dataset cycle")

    monkeypatch.setattr(
        bootstrap_loop_module, "_select_active_evaluator_name", _unexpected_select
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=initialize_bootstrap_run_state(),
        now_unix_s=200.0,
    )

    assert next_state.active_evaluator_name is None
    assert next_state.latest_model_bundle_paths is None


def test_saved_dataset_batches_after_cycle(tmp_path: Path) -> None:
    """A saved rows artifact from the loop should load as a trainable dataset."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    assert state.latest_rows_path is not None
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(
            file_name=str(paths.resolve_work_dir_path(state.latest_rows_path))
        )
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    assert batch.get_input_layer().ndim == 2
    assert batch.get_target_value().shape == (1, 1)
    assert batch.get_input_layer().dtype == torch.float32
    assert batch.get_target_value().dtype == torch.float32


def test_multi_evaluator_failure_propagates_without_history_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one evaluator fails, the whole cycle should fail without a partial event."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    call_count = 0
    real_train = bootstrap_loop_module.train_morpion_regressor

    def _failing_train(args: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("second evaluator failed")
        return real_train(cast("object", args))

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _failing_train
    )

    with pytest.raises(RuntimeError, match="second evaluator failed"):
        run_one_bootstrap_cycle(
            args=args,
            paths=paths,
            runner=runner,
            run_state=initialize_bootstrap_run_state(),
            now_unix_s=200.0,
        )

    assert load_bootstrap_history(paths.history_jsonl_path) == ()
    assert not paths.latest_status_path.exists()


def test_multi_evaluator_training_supports_distinct_feature_subsets(
    tmp_path: Path,
) -> None:
    """Evaluators that differ only by subset should train without shape mismatches."""
    subset_name_10, feature_names_10 = _feature_subset(10)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                "linear_full": MorpionEvaluatorSpec(
                    name="linear_full",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                ),
                "linear_10": MorpionEvaluatorSpec(
                    name="linear_10",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                    feature_subset_name=subset_name_10,
                    feature_names=feature_names_10,
                ),
            }
        ),
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    full_bundle_path = paths.resolve_work_dir_path(
        state.latest_model_bundle_paths["linear_full"]
    )
    subset_bundle_path = paths.resolve_work_dir_path(
        state.latest_model_bundle_paths["linear_10"]
    )
    assert full_bundle_path is not None
    assert subset_bundle_path is not None

    _full_model, full_args, full_manifest = load_morpion_model_bundle(full_bundle_path)
    _subset_model, subset_args, subset_manifest = load_morpion_model_bundle(
        subset_bundle_path
    )

    assert full_args.input_dim == len(MORPION_CANONICAL_FEATURE_NAMES)
    assert subset_args.input_dim == len(feature_names_10)
    assert full_manifest.input_dim == len(MORPION_CANONICAL_FEATURE_NAMES)
    assert subset_manifest.input_dim == len(feature_names_10)
    assert subset_args.feature_names == feature_names_10
    assert subset_manifest.feature_names == feature_names_10


def test_canonical_family_trains_all_eight_evaluators_and_persists_subset_metadata(
    tmp_path: Path,
) -> None:
    """One bootstrap cycle should train and persist the canonical 8-model family."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    history = load_bootstrap_history(paths.history_jsonl_path)
    family = canonical_morpion_evaluator_family_config()

    assert state.latest_model_bundle_paths is not None
    assert set(state.latest_model_bundle_paths) == set(family.evaluators)
    assert state.active_evaluator_name in family.evaluators
    assert len(history) == 1
    assert set(history[0].evaluators) == set(family.evaluators)

    selected_bundle_path = paths.resolve_work_dir_path(
        state.latest_model_bundle_paths[state.active_evaluator_name]
    )
    assert selected_bundle_path is not None
    selected_evaluator = load_morpion_evaluator_from_model_bundle(selected_bundle_path)
    selected_model, selected_args, selected_manifest = load_morpion_model_bundle(
        selected_bundle_path
    )
    assert selected_model is not None
    assert selected_evaluator.feature_converter.input_dim == selected_args.input_dim
    assert selected_manifest.input_dim == selected_args.input_dim

    for evaluator_name, spec in family.evaluators.items():
        bundle_path = paths.resolve_work_dir_path(
            state.latest_model_bundle_paths[evaluator_name]
        )
        assert bundle_path is not None and bundle_path.is_dir()
        _model, loaded_args, loaded_manifest = load_morpion_model_bundle(bundle_path)
        assert loaded_args.feature_subset_name == spec.feature_subset_name
        assert loaded_args.feature_names == spec.feature_names
        assert loaded_args.input_dim == len(spec.feature_names)
        assert loaded_manifest.feature_subset_name == spec.feature_subset_name
        assert loaded_manifest.feature_names == spec.feature_names
        assert loaded_manifest.input_dim == len(spec.feature_names)
