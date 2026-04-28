"""Tests for Morpion bootstrap live control behavior."""
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
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MissingForcedMorpionEvaluatorBundleError,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionBootstrapRuntimeControl,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnknownForcedMorpionEvaluatorError,
    UnsupportedMorpionRuntimeReconfigurationError,
    apply_control_to_args,
    bootstrap_config_from_args,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
    load_bootstrap_history,
    run_morpion_bootstrap_loop,
    run_one_bootstrap_cycle,
    save_bootstrap_control,
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
        is_terminal=True,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-control-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


class FakeMorpionSearchRunner:
    """Deterministic runner used to verify per-cycle control reload behavior."""

    def __init__(
        self,
        *,
        tree_sizes: tuple[int, ...],
        target_values: tuple[float, ...],
        control_path: Path | None = None,
        control_after_first_grow: MorpionBootstrapControl | None = None,
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._cycle_index = -1
        self._control_path = control_path
        self._control_after_first_grow = control_after_first_grow
        self.load_calls: list[tuple[str | None, str | None]] = []
        self.runtime_config_calls: list[
            MorpionBootstrapEffectiveRuntimeConfig | None
        ] = []
        self.grow_calls: list[int] = []

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Record the latest tree/model inputs used to initialize the runner."""
        _ = reevaluate_tree
        self.load_calls.append(
            (
                None if tree_snapshot_path is None else str(tree_snapshot_path),
                None if model_bundle_path is None else str(model_bundle_path),
            )
        )
        self.runtime_config_calls.append(effective_runtime_config)

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner and optionally rewrite the control file."""
        self.grow_calls.append(max_growth_steps)
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1
        if (
            self._cycle_index == 0
            and self._control_path is not None
            and self._control_after_first_grow is not None
        ):
            save_bootstrap_control(self._control_after_first_grow, self._control_path)
            self._control_after_first_grow = None

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
    """Patch training so evaluator selection is deterministic while bundles still exist."""
    real_train = bootstrap_loop_module.train_morpion_regressor

    def _patched_train(train_args: object) -> object:
        _model, metrics = real_train(train_args)
        evaluator_name = Path(
            str(
                cast("bootstrap_loop_module.MorpionTrainingArgs", train_args).output_dir
            )
        ).name
        metrics["final_loss"] = loss_by_evaluator_name[evaluator_name]
        return _model, metrics

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _patched_train
    )


def test_control_roundtrip(tmp_path: Path) -> None:
    """Bootstrap control files should round-trip through JSON unchanged."""
    control = MorpionBootstrapControl(
        max_growth_steps_per_cycle=7,
        max_rows=13,
        use_backed_up_value=False,
        save_after_seconds=12.5,
        save_after_tree_growth_factor=1.5,
        force_evaluator="linear",
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=256),
    )
    control_path = tmp_path / "control.json"

    save_bootstrap_control(control, control_path)

    assert load_bootstrap_control(control_path) == control


def test_apply_control_to_args(tmp_path: Path) -> None:
    """Only non-None control fields should override the base bootstrap args."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        max_rows=11,
        use_backed_up_value=True,
        save_after_seconds=10.0,
        save_after_tree_growth_factor=2.0,
    )
    control = MorpionBootstrapControl(
        max_growth_steps_per_cycle=9,
        max_rows=None,
        use_backed_up_value=False,
        save_after_seconds=None,
        save_after_tree_growth_factor=3.0,
    )

    effective_args = apply_control_to_args(args, control)

    assert effective_args.max_growth_steps_per_cycle == 9
    assert effective_args.max_rows == 11
    assert effective_args.use_backed_up_value is False
    assert effective_args.save_after_seconds == 10.0
    assert effective_args.save_after_tree_growth_factor == 3.0


def test_runtime_control_parsing_tolerates_malformed_fields(tmp_path: Path) -> None:
    """Malformed runtime control fields should be ignored instead of breaking the loop."""
    control_path = tmp_path / "control.json"
    control_path.write_text(
        '{"runtime": {"tree_branch_limit": "nope"}, "max_rows": 7}\n',
        encoding="utf-8",
    )

    control = load_bootstrap_control(control_path)

    assert control.max_rows == 7
    assert control.runtime.tree_branch_limit is None


def test_effective_runtime_config_derivation(tmp_path: Path) -> None:
    """Runtime config derivation should use persisted defaults and control overrides."""
    config = bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=tmp_path, tree_branch_limit=96)
    )

    assert effective_runtime_config_from_config_and_control(
        config,
        MorpionBootstrapControl(),
    ) == MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=96)
    assert effective_runtime_config_from_config_and_control(
        config,
        MorpionBootstrapControl(
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=256)
        ),
    ) == MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=256)


def test_loop_applies_control_between_cycles(tmp_path: Path) -> None:
    """Control changes written during one cycle should apply on the next cycle."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 20),
        target_values=(1.25, -0.5),
        control_path=paths.control_path,
        control_after_first_grow=MorpionBootstrapControl(
            max_growth_steps_per_cycle=9,
            max_rows=3,
            use_backed_up_value=False,
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64),
        ),
    )

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=2)
    history = load_bootstrap_history(paths.history_jsonl_path)

    assert runner.grow_calls == [5, 9]
    assert runner.runtime_config_calls == [
        MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        ),
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64),
    ]
    assert final_state.metadata[BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY] == {
        "force_evaluator": None,
        "max_growth_steps_per_cycle": 9,
        "max_rows": 3,
        "runtime": {"tree_branch_limit": 64},
        "save_after_seconds": None,
        "save_after_tree_growth_factor": None,
        "use_backed_up_value": False,
    }
    assert final_state.metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] == {
        "tree_branch_limit": 64
    }
    assert final_state.metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "tree_branch_limit": 64
    }
    assert isinstance(
        final_state.metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY],
        str,
    )
    assert history[-1].metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] == {
        "tree_branch_limit": 64
    }
    assert history[-1].metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "tree_branch_limit": 64
    }


def test_runtime_control_widening_fails_loudly(tmp_path: Path) -> None:
    """Widening tree_branch_limit on an existing tree should fail explicitly."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 20),
        target_values=(1.25, -0.5),
        control_path=paths.control_path,
        control_after_first_grow=MorpionBootstrapControl(
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=256)
        ),
    )

    with pytest.raises(UnsupportedMorpionRuntimeReconfigurationError):
        run_morpion_bootstrap_loop(args, runner, max_cycles=2)


def test_runtime_reconfiguration_allows_fresh_run_with_any_limit() -> None:
    """A fresh run with no persisted runtime config should accept any limit."""
    bootstrap_loop_module._validate_runtime_reconfiguration(
        previous_effective_runtime_config=None,
        effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=512
        ),
    )


def test_runtime_reconfiguration_allows_same_limit_on_resume() -> None:
    """A resumed tree may keep the same runtime branch limit."""
    previous_config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64)

    bootstrap_loop_module._validate_runtime_reconfiguration(
        previous_effective_runtime_config=previous_config,
        effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=64
        ),
    )


def test_runtime_reconfiguration_allows_lower_limit_on_resume() -> None:
    """A resumed tree may tighten the runtime branch limit."""
    previous_config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64)

    bootstrap_loop_module._validate_runtime_reconfiguration(
        previous_effective_runtime_config=previous_config,
        effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=32
        ),
    )


def test_runtime_reconfiguration_rejects_higher_limit_on_persisted_runtime() -> None:
    """A persisted runtime config should reject widening, even with legacy cycles."""
    previous_config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64)

    with pytest.raises(UnsupportedMorpionRuntimeReconfigurationError):
        bootstrap_loop_module._validate_runtime_reconfiguration(
            previous_effective_runtime_config=previous_config,
            effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
                tree_branch_limit=128
            ),
        )


def test_force_evaluator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A forced evaluator should override auto-selection and next-cycle bundle load."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.7, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_control(
        MorpionBootstrapControl(force_evaluator="linear"),
        paths.control_path,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10, 20), target_values=(1.25, -0.5))

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=2)
    history = load_bootstrap_history(paths.history_jsonl_path)

    assert final_state.active_evaluator_name == "linear"
    assert runner.load_calls[1][1] == str(
        paths.resolve_work_dir_path("models/generation_000001/linear")
    )
    assert history[-1].metadata["forced_evaluator"] == "linear"


def test_invalid_forced_evaluator_fails_loudly(tmp_path: Path) -> None:
    """Unknown forced evaluator names should fail clearly."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_control(
        MorpionBootstrapControl(force_evaluator="ghost"),
        paths.control_path,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    with pytest.raises(UnknownForcedMorpionEvaluatorError):
        run_morpion_bootstrap_loop(args, runner, max_cycles=1)


def test_missing_forced_evaluator_bundle_fails_loudly(tmp_path: Path) -> None:
    """Forced restore should fail when the requested saved bundle is unavailable."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        evaluators_config=_multi_evaluator_config(),
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    run_state = MorpionBootstrapRunState(
        generation=1,
        cycle_index=0,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths={"linear": "models/generation_000001/linear"},
        active_evaluator_name="linear",
        tree_size_at_last_save=10,
        last_save_unix_s=0.0,
    )

    with pytest.raises(MissingForcedMorpionEvaluatorBundleError):
        run_one_bootstrap_cycle(
            args=args,
            paths=MorpionBootstrapPaths.from_work_dir(tmp_path),
            runner=runner,
            run_state=run_state,
            control=MorpionBootstrapControl(force_evaluator="mlp"),
        )
