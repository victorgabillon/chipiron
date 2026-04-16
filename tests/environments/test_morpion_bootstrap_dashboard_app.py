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
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRuntimeControl,
    MorpionBootstrapRunState,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    TreeDepthDistributionRow,
    bootstrap_config_from_args,
    load_bootstrap_config,
    save_bootstrap_config,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.dashboard_app import (
    _selected_child_node_id_for_branch,
    _applied_runtime_control,
    _baseline_tree_branch_limit,
    _build_next_control,
    _configured_evaluator_names,
    _dataset_status_summary,
    _effective_runtime_config,
    _effective_state_summary,
    _effective_runtime_hash,
    _evaluator_set_summary,
    _evaluator_control_status_summary,
    _force_evaluator_options,
    _format_force_evaluator_option,
    _format_force_evaluator_state,
    _render_launcher_command_text,
    _format_optional_runtime_override,
    _format_value,
    _has_pending_control_changes,
    _is_stale_forced_evaluator,
    _load_applied_control,
    _pending_control_fields,
    _pending_control_sections,
    _runtime_status_summary,
    _scheduling_status_summary,
    _tree_inspector_child_rows,
    _tree_structure_rows,
    _tree_branch_limit_input_value,
)
from chipiron.environments.morpion.bootstrap.tree_inspector import (
    MorpionBootstrapChildSummary,
    MorpionBootstrapTreeInspectorSnapshot,
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


def test_runtime_metadata_helpers_are_tolerant() -> None:
    """Dashboard runtime metadata helpers should tolerate absent metadata."""
    run_state = MorpionBootstrapRunState(
        generation=0,
        cycle_index=-1,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
        metadata={},
    )

    assert _applied_runtime_control(run_state) == MorpionBootstrapRuntimeControl()
    assert _effective_runtime_config(run_state) is None
    assert _effective_runtime_hash(run_state) is None


def test_runtime_metadata_helpers_read_present_values() -> None:
    """Dashboard runtime metadata helpers should read applied and effective runtime state."""
    run_state = MorpionBootstrapRunState(
        generation=1,
        cycle_index=0,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
        metadata={
            BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY: {"tree_branch_limit": 64},
            BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY: {"tree_branch_limit": 64},
            BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY: "hash-64",
        },
    )

    assert _applied_runtime_control(run_state) == MorpionBootstrapRuntimeControl(
        tree_branch_limit=64
    )
    assert _effective_runtime_config(run_state) == MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=64
    )
    assert _effective_runtime_hash(run_state) == "hash-64"


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

    configured_evaluator_names = _configured_evaluator_names(
        load_bootstrap_config(paths.bootstrap_config_path)
    )

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


def test_pending_changes_helper_covers_runtime_control() -> None:
    """Runtime-only control changes should still be treated as pending."""
    applied_control = MorpionBootstrapControl(
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64)
    )

    assert not _has_pending_control_changes(applied_control, applied_control)
    assert _has_pending_control_changes(
        MorpionBootstrapControl(
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=96)
        ),
        applied_control,
    )


def test_tree_structure_rows_render_depth_counts_in_order() -> None:
    """Dashboard tree-structure helper should expose sorted per-depth rows."""
    rows = _tree_structure_rows(
        (
            TreeDepthDistributionRow(depth=0, num_nodes=1, cumulative_nodes=1),
            TreeDepthDistributionRow(depth=1, num_nodes=5, cumulative_nodes=6),
            TreeDepthDistributionRow(depth=2, num_nodes=3, cumulative_nodes=9),
        )
    )

    assert rows == [
        {"depth": 0, "num_nodes": 1, "cumulative_nodes": 1},
        {"depth": 1, "num_nodes": 5, "cumulative_nodes": 6},
        {"depth": 2, "num_nodes": 3, "cumulative_nodes": 9},
    ]


def test_pending_control_helpers_cover_expected_categories() -> None:
    """Pending helper outputs should be stable across representative change mixes."""
    applied = MorpionBootstrapControl()

    assert _pending_control_fields(applied, applied) == ()
    assert _pending_control_sections(applied, applied) == ()

    dataset_only = MorpionBootstrapControl(max_rows=10)
    assert _pending_control_fields(dataset_only, applied) == ("max_rows",)
    assert _pending_control_sections(dataset_only, applied) == ("dataset",)

    runtime_only = MorpionBootstrapControl(
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64)
    )
    assert _pending_control_fields(runtime_only, applied) == (
        "runtime.tree_branch_limit",
    )
    assert _pending_control_sections(runtime_only, applied) == ("runtime",)

    forced_only = MorpionBootstrapControl(force_evaluator="mlp")
    assert _pending_control_fields(forced_only, applied) == ("force_evaluator",)
    assert _pending_control_sections(forced_only, applied) == (
        "evaluator selection",
    )

    mixed = MorpionBootstrapControl(
        max_rows=10,
        save_after_seconds=5.0,
        force_evaluator="mlp",
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64),
    )
    assert _pending_control_fields(mixed, applied) == (
        "max_rows",
        "save_after_seconds",
        "force_evaluator",
        "runtime.tree_branch_limit",
    )
    assert _pending_control_sections(mixed, applied) == (
        "dataset",
        "scheduling",
        "evaluator selection",
        "runtime",
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
    ).runtime == MorpionBootstrapRuntimeControl(tree_branch_limit=64)


def test_tree_branch_limit_input_value_prefers_override_then_baseline_then_default() -> None:
    """Dashboard runtime input default should prefer override, then baseline, then constant."""
    assert _tree_branch_limit_input_value(
        runtime_control=MorpionBootstrapRuntimeControl(tree_branch_limit=96),
        baseline_tree_branch_limit=64,
    ) == 96
    assert _tree_branch_limit_input_value(
        runtime_control=MorpionBootstrapRuntimeControl(),
        baseline_tree_branch_limit=64,
    ) == 64
    assert _tree_branch_limit_input_value(
        runtime_control=MorpionBootstrapRuntimeControl(),
        baseline_tree_branch_limit=None,
    ) == DEFAULT_MORPION_TREE_BRANCH_LIMIT


def test_baseline_tree_branch_limit_uses_config_or_default(tmp_path: Path) -> None:
    """Dashboard baseline runtime value should come from config when present."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    assert _baseline_tree_branch_limit(None) == DEFAULT_MORPION_TREE_BRANCH_LIMIT

    args = MorpionBootstrapArgs(work_dir=tmp_path, tree_branch_limit=96)
    save_bootstrap_config(bootstrap_config_from_args(args), paths.bootstrap_config_path)

    assert _baseline_tree_branch_limit(load_bootstrap_config(paths.bootstrap_config_path)) == 96


def test_stale_force_evaluator_helpers() -> None:
    """Force-evaluator formatting should distinguish configured, stale, and empty options."""
    configured = ("linear", "mlp")

    assert not _is_stale_forced_evaluator("linear", configured)
    assert _is_stale_forced_evaluator("old-model", configured)
    assert _is_stale_forced_evaluator("old-model", ())
    assert not _is_stale_forced_evaluator(None, configured)

    assert _format_force_evaluator_option("") == "No configured evaluators"
    assert _format_force_evaluator_option(
        "linear",
        configured_evaluator_names=configured,
    ) == "linear"
    assert _format_force_evaluator_option(
        "old-model",
        configured_evaluator_names=configured,
    ) == "old-model (stale / not configured)"
    assert _format_force_evaluator_option(
        "old-model",
        configured_evaluator_names=(),
    ) == "old-model (stale / not configured)"
    assert _format_force_evaluator_state(
        None,
        configured_evaluator_names=configured,
    ) == "auto"
    assert _format_force_evaluator_state(
        "old-model",
        configured_evaluator_names=configured,
    ) == "old-model (stale / not configured)"


def test_section_status_summaries_are_stable(tmp_path: Path) -> None:
    """Dataset, scheduling, evaluator, and runtime summaries should be deterministic."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_rows=50,
        use_backed_up_value=True,
        max_growth_steps_per_cycle=30,
        save_after_seconds=12.0,
        save_after_tree_growth_factor=1.5,
        tree_branch_limit=96,
        evaluators_config=_multi_evaluator_config(),
    )
    config = bootstrap_config_from_args(args)
    current = MorpionBootstrapControl(
        max_rows=40,
        use_backed_up_value=False,
        max_growth_steps_per_cycle=25,
        save_after_seconds=9.0,
        save_after_tree_growth_factor=1.2,
        force_evaluator="stale-model",
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64),
    )
    applied = MorpionBootstrapControl(
        max_rows=45,
        use_backed_up_value=True,
        max_growth_steps_per_cycle=20,
        save_after_seconds=10.0,
        save_after_tree_growth_factor=1.4,
        force_evaluator="mlp",
        runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=80),
    )

    assert _dataset_status_summary(config, current, applied) == {
        "max_rows": {
            "baseline": 50,
            "current_override": 40,
            "applied_override": 45,
            "effective": 45,
        },
        "use_backed_up_value": {
            "baseline": True,
            "current_override": False,
            "applied_override": True,
            "effective": True,
        },
    }
    assert _scheduling_status_summary(config, current, applied) == {
        "max_growth_steps_per_cycle": {
            "baseline": 30,
            "current_override": 25,
            "applied_override": 20,
            "effective": 20,
        },
        "save_after_seconds": {
            "baseline": 12.0,
            "current_override": 9.0,
            "applied_override": 10.0,
            "effective": 10.0,
        },
        "save_after_tree_growth_factor": {
            "baseline": 1.5,
            "current_override": 1.2,
            "applied_override": 1.4,
            "effective": 1.4,
        },
    }
    assert _evaluator_control_status_summary(
        control=current,
        applied_control=applied,
        configured_evaluator_names=("linear", "mlp"),
    ) == {
        "selection_mode": {
            "baseline": "auto",
            "current_override": "forced",
            "applied_override": "forced",
            "effective": "forced",
        },
        "forced_evaluator": {
            "baseline": None,
            "current_override": "stale-model",
            "applied_override": "mlp",
            "effective": "mlp",
        },
        "current_force_evaluator_is_stale": True,
        "applied_force_evaluator_is_stale": False,
    }
    assert _runtime_status_summary(
        baseline_tree_branch_limit=96,
        current_runtime_control=current.runtime,
        applied_runtime_control=applied.runtime,
        effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=80
        ),
        effective_runtime_hash="hash-80",
    ) == {
        "tree_branch_limit": {
            "baseline": 96,
            "current_override": 64,
            "applied_override": 80,
            "effective": 80,
        },
        "effective_runtime_hash": "hash-80",
    }


def test_effective_state_summary_handles_empty_and_populated_state() -> None:
    """Effective-state summary should stay stable for empty and populated run state."""
    empty_run_state = MorpionBootstrapRunState(
        generation=0,
        cycle_index=-1,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
        metadata={},
    )

    empty_summary = _effective_state_summary(
        run_summary=type("Summary", (), {"latest_active_evaluator_name": None})(),
        run_state=empty_run_state,
        current_control=MorpionBootstrapControl(),
        baseline_tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        effective_runtime_config=None,
        latest_dataset_rows=None,
        pending_changes=False,
        configured_evaluator_names=("linear",),
    )
    assert empty_summary == {
        "active_evaluator": None,
        "forced_evaluator_request": None,
        "forced_evaluator_request_label": "auto",
        "baseline_tree_branch_limit": DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        "effective_tree_branch_limit": None,
        "runtime_override_status": "unset",
        "evaluator_set_label": "custom (1 evaluators)",
        "configured_evaluator_count": 1,
        "configured_evaluator_names": ("linear",),
        "is_canonical_evaluator_family": False,
        "latest_dataset_rows": None,
        "control_pending_application": False,
    }

    populated_run_state = MorpionBootstrapRunState(
        generation=2,
        cycle_index=4,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name="linear",
        tree_size_at_last_save=12,
        last_save_unix_s=None,
        metadata={
            BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY: {"tree_branch_limit": 64},
        },
    )
    populated_summary = _effective_state_summary(
        run_summary=type("Summary", (), {"latest_active_evaluator_name": "mlp"})(),
        run_state=populated_run_state,
        current_control=MorpionBootstrapControl(
            force_evaluator="old-model",
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64),
        ),
        baseline_tree_branch_limit=96,
        effective_runtime_config=MorpionBootstrapEffectiveRuntimeConfig(
            tree_branch_limit=64
        ),
        latest_dataset_rows=123,
        pending_changes=True,
        configured_evaluator_names=("linear", "mlp"),
    )
    assert populated_summary == {
        "active_evaluator": "mlp",
        "forced_evaluator_request": "old-model",
        "forced_evaluator_request_label": "old-model (stale / not configured)",
        "baseline_tree_branch_limit": 96,
        "effective_tree_branch_limit": 64,
        "runtime_override_status": "set",
        "evaluator_set_label": "custom (2 evaluators)",
        "configured_evaluator_count": 2,
        "configured_evaluator_names": ("linear", "mlp"),
        "is_canonical_evaluator_family": False,
        "latest_dataset_rows": 123,
        "control_pending_application": True,
    }


def test_evaluator_set_summary_detects_canonical_family() -> None:
    """Dashboard evaluator-set summary should detect the canonical family exactly."""
    assert _evaluator_set_summary(
        (
            "mlp_41",
            "linear_10",
            "linear_5",
            "mlp_5",
            "linear_20",
            "mlp_10",
            "linear_41",
            "mlp_20",
        )
    ) == {
        "label": "canonical 8-model family",
        "count": 8,
        "configured_evaluator_names": (
            "linear_10",
            "linear_20",
            "linear_41",
            "linear_5",
            "mlp_10",
            "mlp_20",
            "mlp_41",
            "mlp_5",
        ),
        "is_canonical_family": True,
    }


def test_evaluator_set_summary_labels_custom_family() -> None:
    """Dashboard evaluator-set summary should label non-canonical sets as custom."""
    assert _evaluator_set_summary(("linear", "mlp")) == {
        "label": "custom (2 evaluators)",
        "count": 2,
        "configured_evaluator_names": ("linear", "mlp"),
        "is_canonical_family": False,
    }


def test_render_launcher_command_text_joins_parts() -> None:
    """Launcher command rendering should stay stable for the run-control panel."""
    assert _render_launcher_command_text(("python", "-m", "pkg", "--work-dir", "/tmp/run")) == (
        "python -m pkg --work-dir /tmp/run"
    )


def test_format_helpers() -> None:
    """Formatting helpers should keep absent values explicit in the UI."""
    assert _format_value(None) == "n/a"
    assert _format_value(7) == "7"
    assert _format_optional_runtime_override(None) == "unset"
    assert _format_optional_runtime_override(64) == "64"
    assert _format_force_evaluator_option("") == "No configured evaluators"
    assert _format_force_evaluator_option("mlp") == "mlp (stale / not configured)"


def test_tree_inspector_child_rows_are_stable() -> None:
    """Dashboard child-row formatting should preserve display-value priority fields."""
    snapshot = MorpionBootstrapTreeInspectorSnapshot(
        checkpoint_path=None,
        checkpoint_source=None,
        root_node_id="0",
        selected_node_id="0",
        status_message=None,
        error_message=None,
        selection_warning=None,
        node_summary=None,
        child_summaries=(
            MorpionBootstrapChildSummary(
                branch_label="(0,-1,2,3)",
                child_node_id="1",
                visit_count=None,
                is_terminal=False,
                is_exact=False,
                direct_value_scalar=0.2,
                backed_up_value_scalar=0.5,
                display_value_scalar=0.5,
            ),
        ),
        state_view=None,
        local_tree_view=None,
    )

    assert _tree_inspector_child_rows(snapshot) == [
        {
            "branch": "(0,-1,2,3)",
            "child_node_id": "1",
            "display_value": 0.5,
            "backed_up_value": 0.5,
            "direct_value": 0.2,
            "visit_count": None,
            "is_exact": False,
            "is_terminal": False,
        }
    ]


def test_selected_child_node_id_for_branch_returns_matching_child() -> None:
    """Dashboard child navigation helper should resolve the expanded child node id."""
    child_summaries = (
        MorpionBootstrapChildSummary(
            branch_label="a",
            child_node_id="1",
            visit_count=None,
            is_terminal=False,
            is_exact=False,
            direct_value_scalar=0.1,
            backed_up_value_scalar=None,
            display_value_scalar=0.1,
        ),
        MorpionBootstrapChildSummary(
            branch_label="b",
            child_node_id=None,
            visit_count=None,
            is_terminal=None,
            is_exact=None,
            direct_value_scalar=None,
            backed_up_value_scalar=None,
            display_value_scalar=None,
        ),
    )

    assert _selected_child_node_id_for_branch(child_summaries, "a") == "1"
    assert _selected_child_node_id_for_branch(child_summaries, "b") is None
    assert _selected_child_node_id_for_branch(child_summaries, "missing") is None
