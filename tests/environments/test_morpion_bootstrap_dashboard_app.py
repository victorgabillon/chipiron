"""Tests for pure Morpion bootstrap dashboard-app helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from chipiron.environments.morpion.bootstrap import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    ActiveEvaluatorTimeSeriesPoint,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionBootstrapRuntimeControl,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    OptionalFloatTimeSeriesPoint,
    TreeDepthDistributionRow,
    bootstrap_config_from_args,
    load_bootstrap_config,
    save_bootstrap_config,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    cached_build_morpion_bootstrap_dashboard_data,
    cached_dashboard_data_freshness_tokens,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    downsample_loss_series_by_name as _downsample_loss_series_by_name,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    downsample_series as _downsample_series,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.certified_record import (
    render_current_certified_record_board_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.disk_usage import (
    render_disk_usage_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
    diagnostic_examples_rows as _diagnostic_examples_rows,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
    load_latest_evaluator_training_diagnostics_for_dashboard as _load_latest_evaluator_training_diagnostics_for_dashboard,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
    render_evaluator_training_diagnostics_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.linoo import (
    linoo_selection_table_rows as _linoo_selection_table_rows,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.linoo import (
    render_linoo_selection_table,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    observability_summary_from_metadata as _observability_summary_from_metadata,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    render_observability_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
    render_plot,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.record_status import (
    render_record_status_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.run_control import (
    render_launcher_command_text as _render_launcher_command_text,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.run_control import (
    render_run_control_section,
    render_run_control_state,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    applied_runtime_control as _applied_runtime_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    baseline_tree_branch_limit as _baseline_tree_branch_limit,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    build_next_control as _build_next_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    configured_evaluator_names as _configured_evaluator_names,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    dataset_status_summary as _dataset_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_runtime_config as _effective_runtime_config,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_runtime_hash as _effective_runtime_hash,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_state_summary as _effective_state_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    evaluator_control_status_summary as _evaluator_control_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    evaluator_set_summary as _evaluator_set_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    force_evaluator_options as _force_evaluator_options,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    format_force_evaluator_option as _format_force_evaluator_option,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    format_force_evaluator_state as _format_force_evaluator_state,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    has_pending_control_changes as _has_pending_control_changes,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    is_stale_forced_evaluator as _is_stale_forced_evaluator,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_applied_control as _load_applied_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_fields as _pending_control_fields,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_sections as _pending_control_sections,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_dataset_control_section,
    render_evaluator_control_section,
    render_runtime_control_section,
    render_scheduling_control_section,
    render_status_layers,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    runtime_status_summary as _runtime_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    scheduling_status_summary as _scheduling_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    tree_branch_limit_input_value as _tree_branch_limit_input_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
    render_tree_inspector_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
    selected_child_node_id_for_branch as _selected_child_node_id_for_branch,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
    tree_inspector_child_rows as _tree_inspector_child_rows,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
    render_tree_structure_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
    tree_structure_rows as _tree_structure_rows,
)
from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
    MorpionBootstrapChildSummary,
    MorpionBootstrapTreeInspectorSnapshot,
)
from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
    MorpionEvaluatorDiagnosticExample,
    MorpionEvaluatorTrainingDiagnostics,
    save_evaluator_training_diagnostics,
)
from chipiron.environments.morpion.bootstrap.linoo_selection_table import (
    LinooSelectionTable,
    LinooSelectionTableRow,
)


def test_dashboard_forward_imports_resolve_owner_modules() -> None:
    """Dashboard APIs should resolve from the owner package modules."""
    import chipiron.environments.morpion.bootstrap.dashboard as dashboard_pkg
    import chipiron.environments.morpion.bootstrap.dashboard.app as dashboard_app_new
    import chipiron.environments.morpion.bootstrap.dashboard.data_cache as dashboard_data_cache_new
    import chipiron.environments.morpion.bootstrap.dashboard.formatting as dashboard_formatting_new
    import chipiron.environments.morpion.bootstrap.dashboard.history_view as history_view_new
    import chipiron.environments.morpion.bootstrap.dashboard.plot as dashboard_plot_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.certified_record as dashboard_certified_record_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.disk_usage as dashboard_disk_usage_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics as dashboard_evaluator_diagnostics_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.linoo as dashboard_linoo_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.observability as dashboard_observability_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.plot as dashboard_plot_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.record_status as dashboard_record_status_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.run_control as dashboard_run_control_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers as dashboard_status_layers_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector as dashboard_tree_inspector_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure as dashboard_tree_structure_section_new
    import chipiron.environments.morpion.bootstrap.dashboard.streamlit_entry as streamlit_new
    import chipiron.environments.morpion.bootstrap.dashboard.tree_inspector as tree_inspector_new

    assert dashboard_pkg.run_dashboard_app is dashboard_app_new.run_dashboard_app
    assert callable(streamlit_new.main)
    assert (
        dashboard_data_cache_new.cached_build_morpion_bootstrap_dashboard_data
        is cached_build_morpion_bootstrap_dashboard_data
    )
    assert (
        dashboard_data_cache_new.cached_dashboard_data_freshness_tokens
        is cached_dashboard_data_freshness_tokens
    )
    assert dashboard_formatting_new.downsample_series is _downsample_series
    assert dashboard_formatting_new.format_value is _format_value
    assert callable(dashboard_plot_new.plot_tree_size)
    assert dashboard_disk_usage_section_new.render_disk_usage_section is (
        render_disk_usage_section
    )
    assert dashboard_observability_section_new.render_observability_section is (
        render_observability_section
    )
    assert dashboard_plot_section_new.render_plot is render_plot
    assert dashboard_run_control_section_new.render_run_control_section is (
        render_run_control_section
    )
    assert dashboard_run_control_section_new.render_run_control_state is (
        render_run_control_state
    )
    assert dashboard_status_layers_section_new.render_status_layers is (
        render_status_layers
    )
    assert dashboard_status_layers_section_new.render_dataset_control_section is (
        render_dataset_control_section
    )
    assert dashboard_status_layers_section_new.render_scheduling_control_section is (
        render_scheduling_control_section
    )
    assert dashboard_status_layers_section_new.render_evaluator_control_section is (
        render_evaluator_control_section
    )
    assert dashboard_status_layers_section_new.render_runtime_control_section is (
        render_runtime_control_section
    )
    assert dashboard_tree_inspector_section_new.render_tree_inspector_section is (
        render_tree_inspector_section
    )
    assert dashboard_tree_structure_section_new.render_tree_structure_section is (
        render_tree_structure_section
    )
    assert dashboard_linoo_section_new.render_linoo_selection_table is (
        render_linoo_selection_table
    )
    assert dashboard_record_status_section_new.render_record_status_section is (
        render_record_status_section
    )
    assert (
        dashboard_evaluator_diagnostics_section_new.render_evaluator_training_diagnostics_section
        is render_evaluator_training_diagnostics_section
    )
    assert (
        dashboard_certified_record_section_new.render_current_certified_record_board_section
        is render_current_certified_record_board_section
    )
    assert (
        history_view_new.build_morpion_bootstrap_dashboard_data.__name__
        == "build_morpion_bootstrap_dashboard_data"
    )
    assert (
        tree_inspector_new.build_morpion_bootstrap_tree_inspector_snapshot.__name__
        == "build_morpion_bootstrap_tree_inspector_snapshot"
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
    assert _effective_runtime_config(
        run_state
    ) == MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64)
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
        configured_names=configured_evaluator_names,
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


def test_observability_summary_derives_ratios_and_fast_path_health() -> None:
    """Dashboard observability summary should derive C5/C6 health fields safely."""
    summary = _observability_summary_from_metadata({
        "tree": {"node_count": 200, "branch_count": 240},
        "memory": {"rss_mb": 1200.0},
        "state_eviction": {
            "compact_payload_count": 180,
            "delta_payload_count": 153,
            "eviction_success_count": 180,
            "eviction_skipped_count": 2,
            "delta_payload_fallback_count": 5,
        },
        "checkpoint": {"total_s": 24.5, "bytes": 1024},
        "training_export": {"total_s": 28.75, "rows_written": 400},
        "training_export_profile": {
            "node_count": 200,
            "state_access_calls": 20,
            "checkpoint_backed_state_handles": 180,
            "reusable_checkpoint_payloads": 180,
            "plain_or_materialized_states": 20,
        },
    })

    assert summary["delta_payload_ratio"] == 0.85
    assert summary["checkpoint_backed_ratio"] == 0.9
    assert summary["materialized_ratio"] == 0.1
    assert summary["export_fast_path_health"] == "good"


def test_observability_summary_handles_missing_and_zero_denominators() -> None:
    """Older status files and zero denominators should not crash summaries."""
    summary = _observability_summary_from_metadata({
        "tree": {"node_count": 0},
        "state_eviction": {"compact_payload_count": 0, "delta_payload_count": 0},
    })

    assert summary["delta_payload_ratio"] is None
    assert summary["checkpoint_backed_ratio"] is None
    assert summary["materialized_ratio"] is None
    assert summary["export_fast_path_health"] == "unknown"


def test_observability_summary_warns_when_export_resolves_most_states() -> None:
    """Dashboard health should flag fast-path regressions."""
    summary = _observability_summary_from_metadata({
        "training_export_profile": {
            "node_count": 100,
            "state_access_calls": 95,
            "plain_or_materialized_states": 5,
        }
    })

    assert summary["export_fast_path_health"] == "warning"


def test_observability_summary_handles_skipped_training_export() -> None:
    """Skipped training exports should remain visible without health errors."""
    summary = _observability_summary_from_metadata({
        "tree": {
            "node_count": 2000,
            "growth_budget_mode": "additional",
            "effective_branch_limit": 3000,
        },
        "training_export": {"status": "skipped", "reason": "config"},
    })

    assert summary["training_export"] == {"status": "skipped", "reason": "config"}
    assert summary["tree"]["growth_budget_mode"] == "additional"
    assert summary["tree"]["effective_branch_limit"] == 3000
    assert summary["export_fast_path_health"] == "unknown"


def test_tree_structure_rows_render_depth_counts_in_order() -> None:
    """Dashboard tree-structure helper should expose sorted per-depth rows."""
    rows = _tree_structure_rows((
        TreeDepthDistributionRow(depth=0, num_nodes=1, cumulative_nodes=1),
        TreeDepthDistributionRow(depth=1, num_nodes=5, cumulative_nodes=6),
        TreeDepthDistributionRow(depth=2, num_nodes=3, cumulative_nodes=9),
    ))

    assert rows == [
        {"depth": 0, "num_nodes": 1, "cumulative_nodes": 1},
        {"depth": 1, "num_nodes": 5, "cumulative_nodes": 6},
        {"depth": 2, "num_nodes": 3, "cumulative_nodes": 9},
    ]


def test_linoo_selection_table_rows_expose_dashboard_columns() -> None:
    """Dashboard should map persisted Linoo rows to display column names."""
    rows = _linoo_selection_table_rows(
        LinooSelectionTable(
            updated_at_utc="2026-04-11T09:00:00Z",
            cycle_index=4,
            generation=6,
            step=7,
            selected_depth=4,
            selected_node_id=1550,
            rows=(
                LinooSelectionTableRow(
                    depth=4,
                    opened=9,
                    frontier=31,
                    deterministic_index=45,
                    weight=0.2,
                    probability=0.4,
                    best_node=1550,
                    best_value=0.38,
                    selected=True,
                ),
            ),
        )
    )

    assert rows == [
        {
            "depth": 4,
            "opened_count": 9,
            "frontier_count": 31,
            "deterministic_index": 45,
            "weight": 0.2,
            "probability": 0.4,
            "best_node_id": 1550,
            "best_direct_value": 0.38,
            "selected": True,
        }
    ]


def test_downsample_series_keeps_small_history_unchanged() -> None:
    """Series shorter than the cap should pass through unchanged."""
    series = (
        ActiveEvaluatorTimeSeriesPoint(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-16T08:00:00Z",
            active_evaluator_name="linear",
        ),
        ActiveEvaluatorTimeSeriesPoint(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-16T09:00:00Z",
            active_evaluator_name="mlp",
        ),
    )

    assert _downsample_series(series, max_points=5) == series


def test_downsample_series_respects_point_cap_and_preserves_endpoints() -> None:
    """Downsampling should preserve the first and last points exactly."""
    series = tuple(range(10))

    downsampled = _downsample_series(series, max_points=4)

    assert len(downsampled) == 4
    assert downsampled[0] == 0
    assert downsampled[-1] == 9


def test_downsample_loss_series_by_name_bounds_each_evaluator_series() -> None:
    """Loss-series downsampling should bound each evaluator independently."""
    linear_series = tuple(
        OptionalFloatTimeSeriesPoint(
            cycle_index=index,
            generation=index + 1,
            timestamp_utc=f"2026-04-16T08:{index:02d}:00Z",
            value=float(index),
        )
        for index in range(8)
    )
    mlp_series = tuple(
        OptionalFloatTimeSeriesPoint(
            cycle_index=index,
            generation=index + 1,
            timestamp_utc=f"2026-04-16T09:{index:02d}:00Z",
            value=float(index) / 10.0,
        )
        for index in range(3)
    )

    downsampled = _downsample_loss_series_by_name(
        {"linear": linear_series, "mlp": mlp_series},
        max_points=4,
    )

    assert len(downsampled["linear"]) == 4
    assert downsampled["linear"][0] == linear_series[0]
    assert downsampled["linear"][-1] == linear_series[-1]
    assert downsampled["mlp"] == mlp_series


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
    assert _pending_control_sections(forced_only, applied) == ("evaluator selection",)

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


def test_tree_branch_limit_input_value_prefers_override_then_baseline_then_default() -> (
    None
):
    """Dashboard runtime input default should prefer override, then baseline, then constant."""
    assert (
        _tree_branch_limit_input_value(
            runtime_control=MorpionBootstrapRuntimeControl(tree_branch_limit=96),
            baseline_limit=64,
        )
        == 96
    )
    assert (
        _tree_branch_limit_input_value(
            runtime_control=MorpionBootstrapRuntimeControl(),
            baseline_limit=64,
        )
        == 64
    )
    assert (
        _tree_branch_limit_input_value(
            runtime_control=MorpionBootstrapRuntimeControl(),
            baseline_limit=None,
        )
        == DEFAULT_MORPION_TREE_BRANCH_LIMIT
    )


def test_baseline_tree_branch_limit_uses_config_or_default(tmp_path: Path) -> None:
    """Dashboard baseline runtime value should come from config when present."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    assert _baseline_tree_branch_limit(None) == DEFAULT_MORPION_TREE_BRANCH_LIMIT

    args = MorpionBootstrapArgs(work_dir=tmp_path, tree_branch_limit=96)
    save_bootstrap_config(bootstrap_config_from_args(args), paths.bootstrap_config_path)

    assert (
        _baseline_tree_branch_limit(load_bootstrap_config(paths.bootstrap_config_path))
        == 96
    )


def test_stale_force_evaluator_helpers() -> None:
    """Force-evaluator formatting should distinguish configured, stale, and empty options."""
    configured = ("linear", "mlp")

    assert not _is_stale_forced_evaluator("linear", configured)
    assert _is_stale_forced_evaluator("old-model", configured)
    assert _is_stale_forced_evaluator("old-model", ())
    assert not _is_stale_forced_evaluator(None, configured)

    assert _format_force_evaluator_option("") == "No configured evaluators"
    assert (
        _format_force_evaluator_option(
            "linear",
            configured_names=configured,
        )
        == "linear"
    )
    assert (
        _format_force_evaluator_option(
            "old-model",
            configured_names=configured,
        )
        == "old-model (stale / not configured)"
    )
    assert (
        _format_force_evaluator_option(
            "old-model",
            configured_names=(),
        )
        == "old-model (stale / not configured)"
    )
    assert (
        _format_force_evaluator_state(
            None,
            configured_names=configured,
        )
        == "auto"
    )
    assert (
        _format_force_evaluator_state(
            "old-model",
            configured_names=configured,
        )
        == "old-model (stale / not configured)"
    )


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
        configured_names=("linear", "mlp"),
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
        baseline_limit=96,
        current_runtime_control=current.runtime,
        applied_runtime=applied.runtime,
        effective_runtime=MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=80),
        runtime_hash="hash-80",
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
        baseline_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        effective_runtime=None,
        latest_dataset_rows=None,
        pending_changes=False,
        configured_names=("linear",),
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
        baseline_limit=96,
        effective_runtime=MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64),
        latest_dataset_rows=123,
        pending_changes=True,
        configured_names=("linear", "mlp"),
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
    assert _evaluator_set_summary((
        "mlp_41",
        "linear_10",
        "linear_5",
        "mlp_5",
        "linear_20",
        "mlp_10",
        "linear_41",
        "mlp_20",
    )) == {
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
    assert _render_launcher_command_text((
        "python",
        "-m",
        "pkg",
        "--work-dir",
        "/tmp/run",
    )) == ("python -m pkg --work-dir /tmp/run")


def test_format_helpers() -> None:
    """Formatting helpers should keep absent values explicit in the UI."""
    assert _format_value(None) == "n/a"
    assert _format_value(7) == "7"
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
            "is_exact": "✖",
            "is_terminal": "✖",
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


def test_dashboard_diagnostics_loader_is_graceful_when_artifacts_are_absent(
    tmp_path: Path,
) -> None:
    """Dashboard diagnostics helpers should tolerate missing run artifacts."""
    assert _load_latest_evaluator_training_diagnostics_for_dashboard(tmp_path) == {}


def test_dashboard_diagnostics_rows_and_loader_use_latest_generation(
    tmp_path: Path,
) -> None:
    """Dashboard diagnostics helpers should expose the newest persisted generation."""
    older = MorpionEvaluatorTrainingDiagnostics(
        generation=4,
        evaluator_name="linear",
        dataset_size=10,
        created_at="2026-04-24T10:00:00Z",
        representative_examples=[],
        worst_examples=[],
        mae_before=None,
        mae_after=0.3,
        max_abs_error_before=None,
        max_abs_error_after=0.6,
    )
    latest = MorpionEvaluatorTrainingDiagnostics(
        generation=5,
        evaluator_name="mlp",
        dataset_size=12,
        created_at="2026-04-24T10:05:00Z",
        representative_examples=[
            MorpionEvaluatorDiagnosticExample(
                row_index=7,
                node_id="node-7",
                state_tag=21,
                depth=3,
                target_value=1.5,
                prediction_before=0.5,
                prediction_after=1.25,
                abs_error_before=1.0,
                abs_error_after=0.25,
            )
        ],
        worst_examples=[],
        mae_before=0.9,
        mae_after=0.2,
        max_abs_error_before=1.0,
        max_abs_error_after=0.25,
    )

    save_evaluator_training_diagnostics(
        older,
        tmp_path / "evaluator_diagnostics" / "generation_000004" / "linear.json",
    )
    save_evaluator_training_diagnostics(
        latest,
        tmp_path / "evaluator_diagnostics" / "generation_000005" / "mlp.json",
    )

    loaded = _load_latest_evaluator_training_diagnostics_for_dashboard(tmp_path)
    rows = _diagnostic_examples_rows(latest.representative_examples)

    assert tuple(loaded) == ("mlp",)
    assert loaded["mlp"] == latest
    assert rows == [
        {
            "row_index": 7,
            "node_id": "node-7",
            "state_tag": 21,
            "depth": 3,
            "target_value": 1.5,
            "prediction_before": 0.5,
            "prediction_after": 1.25,
            "abs_error_before": 1.0,
            "abs_error_after": 0.25,
        }
    ]
