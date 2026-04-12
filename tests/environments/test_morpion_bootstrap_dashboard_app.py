"""Tests for pure Morpion bootstrap dashboard-app helpers."""

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

from chipiron.environments.morpion.bootstrap import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapPaths,
    MorpionBootstrapRuntimeControl,
    MorpionBootstrapRunState,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    bootstrap_config_from_args,
    save_bootstrap_config,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.dashboard_app import (
    _build_next_control,
    _configured_evaluator_names,
    _force_evaluator_options,
    _format_force_evaluator_option,
    _format_value,
    _has_pending_control_changes,
    _load_applied_control,
)


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


def test_load_applied_control_reads_run_state_metadata(tmp_path: Path) -> None:
    """The dashboard should deserialize the last applied control from run state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    applied_control = MorpionBootstrapControl(
        max_rows=11,
        save_after_seconds=12.5,
        force_evaluator="linear",
    )
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
            metadata={
                BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY: {
                    "force_evaluator": "linear",
                    "max_growth_steps_per_cycle": None,
                    "max_rows": 11,
                    "save_after_seconds": 12.5,
                    "save_after_tree_growth_factor": None,
                    "use_backed_up_value": None,
                }
            },
        ),
        paths.run_state_path,
    )

    assert _load_applied_control(paths) == applied_control


def test_configured_force_evaluator_options_come_from_config(tmp_path: Path) -> None:
    """Dashboard force-evaluator choices should come from persisted config names."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_config(
        bootstrap_config_from_args(args),
        paths.bootstrap_config_path,
    )

    configured_evaluator_names = _configured_evaluator_names(paths)

    assert configured_evaluator_names == ("linear", "mlp")
    assert _force_evaluator_options(
        configured_evaluator_names=configured_evaluator_names,
        current_force_evaluator=None,
    ) == ("linear", "mlp")


def test_pending_changes_helper() -> None:
    """Dashboard pending detection should compare current and applied control exactly."""
    applied_control = MorpionBootstrapControl(max_rows=11)

    assert not _has_pending_control_changes(applied_control, applied_control)
    assert _has_pending_control_changes(
        MorpionBootstrapControl(max_rows=12),
        applied_control,
    )


def test_build_next_control_preserves_unset_overrides() -> None:
    """Unchecked dashboard overrides should round-trip back to None."""
    assert _build_next_control(
        override_max_growth_steps_per_cycle=False,
        max_growth_steps_per_cycle=9,
        override_max_rows=True,
        max_rows=17,
        override_use_backed_up_value=False,
        use_backed_up_value=False,
        override_save_after_seconds=False,
        save_after_seconds=30.0,
        override_save_after_tree_growth_factor=True,
        save_after_tree_growth_factor=1.5,
        override_tree_branch_limit=False,
        tree_branch_limit=512,
        force_evaluator_mode="auto",
        force_evaluator="linear",
        current_runtime_control=MorpionBootstrapRuntimeControl(tree_branch_limit=256),
    ) == MorpionBootstrapControl(
        max_growth_steps_per_cycle=None,
        max_rows=17,
        use_backed_up_value=None,
        save_after_seconds=None,
        save_after_tree_growth_factor=1.5,
        force_evaluator=None,
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=None),
    )


def test_build_next_control_applies_runtime_override() -> None:
    """Checked runtime overrides should persist the selected tree branch limit."""
    assert _build_next_control(
        override_max_growth_steps_per_cycle=False,
        max_growth_steps_per_cycle=9,
        override_max_rows=False,
        max_rows=17,
        override_use_backed_up_value=False,
        use_backed_up_value=False,
        override_save_after_seconds=False,
        save_after_seconds=30.0,
        override_save_after_tree_growth_factor=False,
        save_after_tree_growth_factor=1.5,
        override_tree_branch_limit=True,
        tree_branch_limit=64,
        force_evaluator_mode="auto",
        force_evaluator="linear",
        current_runtime_control=MorpionBootstrapRuntimeControl(),
    ).runtime == MorpionBootstrapRuntimeControl(tree_branch_limit=64)


def test_format_helpers() -> None:
    """Formatting helpers should keep absent values explicit in the UI."""
    assert _format_value(None) == "n/a"
    assert _format_value(7) == "7"
    assert _format_force_evaluator_option("") == "No configured evaluators"
    assert _format_force_evaluator_option("mlp") == "mlp"