"""Local Streamlit dashboard for monitoring and controlling Morpion bootstrap runs."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.control import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    bootstrap_control_to_dict,
    load_bootstrap_control,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    cached_build_current_certified_record_board_view as _cached_build_current_certified_record_board_view,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    cached_build_morpion_bootstrap_dashboard_data as _cached_build_morpion_bootstrap_dashboard_data,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    cached_certified_record_board_freshness_tokens as _cached_certified_record_board_freshness_tokens,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    cached_dashboard_data_freshness_tokens as _cached_dashboard_data_freshness_tokens,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    checked_training_status_files_summary,
    loss_series_contains_points,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    control_float_value as _control_float_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    control_number_value as _control_number_value,
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
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    latest_optional_value as _latest_optional_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.plot import (
    plot_active_evaluator,
    plot_certified_record_score,
    plot_dataset_size,
    plot_evaluator_losses,
    plot_tree_size,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.certified_record import (
    render_current_certified_record_board_section as _render_current_certified_record_board_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.disk_usage import (
    render_disk_usage_section as _render_disk_usage_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
    has_known_optional_series_values as _has_known_optional_series_values,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
    render_evaluator_training_diagnostics_section as _render_evaluator_training_diagnostics_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    observability_summary_from_metadata,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    render_observability_section as _render_observability_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
    render_plot as _render_plot,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.record_status import (
    render_record_status_section as _render_record_status_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.run_control import (
    render_run_control_section as _render_run_control_section,
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
    force_evaluator_option_index as _force_evaluator_option_index,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    force_evaluator_options as _force_evaluator_options,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    format_force_evaluator_option as _format_force_evaluator_option,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    has_pending_control_changes as _has_pending_control_changes,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_applied_control as _load_applied_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_bootstrap_config_or_none as _load_bootstrap_config_or_none,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_run_state as _load_run_state,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_fields as _pending_control_fields,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_sections as _pending_control_sections,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_dataset_control_section as _render_dataset_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_effective_state_section as _render_effective_state_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_evaluator_control_section as _render_evaluator_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_pending_changes_section as _render_pending_changes_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_runtime_control_section as _render_runtime_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_scheduling_control_section as _render_scheduling_control_section,
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
    render_tree_inspector_fragment as _render_tree_inspector_fragment,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
    render_tree_structure_section as _render_tree_structure_section,
)


def run_dashboard_app(work_dir: Path) -> None:
    """Render the local Streamlit dashboard for one bootstrap work directory."""
    st = _get_streamlit()
    st.set_page_config(page_title="Morpion Bootstrap Dashboard", layout="wide")

    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    config = _load_bootstrap_config_or_none(paths)
    control = load_bootstrap_control(paths.control_path)
    applied_control = _load_applied_control(paths)
    dashboard_data = _cached_build_morpion_bootstrap_dashboard_data(
        str(paths.work_dir),
        _cached_dashboard_data_freshness_tokens(paths),
    )

    run_state = _load_run_state(paths)
    pending_changes = _has_pending_control_changes(control, applied_control)
    configured_evaluator_names = _configured_evaluator_names(config)
    force_evaluator_options = _force_evaluator_options(
        configured_names=configured_evaluator_names,
        current_force_evaluator=control.force_evaluator,
    )
    baseline_tree_branch_limit = _baseline_tree_branch_limit(config)
    applied_runtime_control = _applied_runtime_control(run_state)
    effective_runtime_config = _effective_runtime_config(run_state)
    effective_runtime_hash = _effective_runtime_hash(run_state)
    tree_branch_limit_input_value = _tree_branch_limit_input_value(
        runtime_control=control.runtime,
        baseline_limit=baseline_tree_branch_limit,
    )
    pending_fields = _pending_control_fields(control, applied_control)
    pending_sections = _pending_control_sections(control, applied_control)
    dataset_summary = _dataset_status_summary(config, control, applied_control)
    scheduling_summary = _scheduling_status_summary(config, control, applied_control)
    evaluator_summary = _evaluator_control_status_summary(
        control=control,
        applied_control=applied_control,
        configured_names=configured_evaluator_names,
    )
    runtime_summary = _runtime_status_summary(
        baseline_limit=baseline_tree_branch_limit,
        current_runtime_control=control.runtime,
        applied_runtime=applied_runtime_control,
        effective_runtime=effective_runtime_config,
        runtime_hash=effective_runtime_hash,
    )
    st.title("Morpion Bootstrap Dashboard")
    st.caption(str(paths.work_dir))
    board_view = _cached_build_current_certified_record_board_view(
        str(paths.work_dir),
        _cached_certified_record_board_freshness_tokens(paths),
    )

    summary = dashboard_data.run_summary
    latest_dataset_rows = _latest_optional_value(dashboard_data.dataset_num_rows)
    status_columns = st.columns(4)
    status_columns[0].metric("Generation", _format_value(summary.latest_generation))
    status_columns[1].metric("Tree Size", _format_value(summary.latest_tree_num_nodes))
    status_columns[2].metric(
        "Active Evaluator",
        _format_value(summary.latest_active_evaluator_name),
    )
    status_columns[3].metric("Dataset Rows", _format_value(latest_dataset_rows))

    st.subheader("Disk Usage")
    _render_disk_usage_section(st=st, summary=dashboard_data.disk_usage_summary)

    st.subheader("Memory / Export Observability")
    _render_observability_section(
        st=st,
        summary=observability_summary_from_metadata(run_state.metadata),
    )

    st.subheader("Record Status")
    _render_record_status_section(
        st=st,
        certified_status=dashboard_data.latest_certified_record_status,
        frontier_status=dashboard_data.latest_frontier_status,
    )

    st.subheader("Current Certified Record Board")
    _render_current_certified_record_board_section(
        st=st,
        board_view=board_view,
    )

    _render_run_control_section(st=st, paths=paths)

    st.subheader("Controls")
    _render_pending_changes_section(
        st=st,
        pending_changes=pending_changes,
        pending_sections=pending_sections,
        pending_fields=pending_fields,
    )
    _render_effective_state_section(
        st=st,
        summary=_effective_state_summary(
            run_summary=summary,
            run_state=run_state,
            current_control=control,
            baseline_limit=baseline_tree_branch_limit,
            effective_runtime=effective_runtime_config,
            latest_dataset_rows=latest_dataset_rows,
            pending_changes=pending_changes,
            configured_names=configured_evaluator_names,
        ),
    )
    st.caption(
        "Unchecked controls inherit the persisted bootstrap config baseline. "
        "Checked controls persist explicit overrides to the control file."
    )

    with st.form("bootstrap-controls"):
        _render_dataset_control_section(st=st, summary=dataset_summary)
        override_max_rows = st.checkbox(
            "Persist explicit override for max rows",
            value=control.max_rows is not None,
        )
        max_rows = st.number_input(
            "Max rows",
            min_value=0,
            value=_control_number_value(control.max_rows),
            disabled=not override_max_rows,
        )
        override_use_backed_up_value = st.checkbox(
            "Persist explicit override for use backed-up value",
            value=control.use_backed_up_value is not None,
        )
        use_backed_up_value = st.checkbox(
            "Use backed-up value",
            value=False
            if control.use_backed_up_value is None
            else control.use_backed_up_value,
            disabled=not override_use_backed_up_value,
        )

        _render_scheduling_control_section(st=st, summary=scheduling_summary)
        override_max_growth_steps_per_cycle = st.checkbox(
            "Persist explicit override for max growth steps per cycle",
            value=control.max_growth_steps_per_cycle is not None,
        )
        max_growth_steps_per_cycle = st.number_input(
            "Max growth steps per cycle",
            min_value=0,
            value=_control_number_value(control.max_growth_steps_per_cycle),
            disabled=not override_max_growth_steps_per_cycle,
        )
        override_save_after_seconds = st.checkbox(
            "Persist explicit override for save after seconds",
            value=control.save_after_seconds is not None,
        )
        save_after_seconds = st.number_input(
            "Save after seconds",
            min_value=0.0,
            value=_control_float_value(control.save_after_seconds),
            step=1.0,
            disabled=not override_save_after_seconds,
        )
        override_save_after_tree_growth_factor = st.checkbox(
            "Persist explicit override for save after tree growth factor",
            value=control.save_after_tree_growth_factor is not None,
        )
        save_after_tree_growth_factor = st.number_input(
            "Save after tree growth factor",
            min_value=0.0,
            value=_control_float_value(
                control.save_after_tree_growth_factor, default=2.0
            ),
            step=0.1,
            disabled=not override_save_after_tree_growth_factor,
        )

        _render_evaluator_control_section(st=st, summary=evaluator_summary)
        force_evaluator_mode = st.radio(
            "Evaluator selection mode",
            options=("auto", "forced"),
            index=0 if control.force_evaluator is None else 1,
            horizontal=True,
            help="Auto inherits normal evaluator selection. Forced persists an explicit evaluator override.",
        )
        selectable_force_evaluator_options = (
            force_evaluator_options if force_evaluator_options else ("",)
        )

        def _format_option(value: str) -> str:
            return _format_force_evaluator_option(
                value,
                configured_names=configured_evaluator_names,
            )

        force_evaluator = st.selectbox(
            "Forced evaluator",
            options=selectable_force_evaluator_options,
            index=_force_evaluator_option_index(
                selectable_force_evaluator_options,
                control.force_evaluator,
            ),
            disabled=(force_evaluator_mode != "forced" or not force_evaluator_options),
            format_func=_format_option,
        )

        _render_runtime_control_section(st=st, summary=runtime_summary)
        override_tree_branch_limit = st.checkbox(
            "Persist explicit override for tree branch limit",
            value=control.runtime.tree_branch_limit is not None,
        )
        tree_branch_limit = st.number_input(
            "Tree branch limit",
            min_value=1,
            value=tree_branch_limit_input_value,
            disabled=not override_tree_branch_limit,
        )

        if st.form_submit_button("Apply changes"):
            next_control = _build_next_control(
                override_max_growth_steps_per_cycle=override_max_growth_steps_per_cycle,
                max_growth_steps_per_cycle=max_growth_steps_per_cycle,
                override_max_rows=override_max_rows,
                max_rows=max_rows,
                override_use_backed_up_value=override_use_backed_up_value,
                use_backed_up_value=use_backed_up_value,
                override_save_after_seconds=override_save_after_seconds,
                save_after_seconds=save_after_seconds,
                override_save_after_tree_growth_factor=override_save_after_tree_growth_factor,
                save_after_tree_growth_factor=save_after_tree_growth_factor,
                override_tree_branch_limit=override_tree_branch_limit,
                tree_branch_limit=tree_branch_limit,
                force_evaluator_mode=force_evaluator_mode,
                force_evaluator=force_evaluator,
            )
            save_bootstrap_control(next_control, paths.control_path)
            st.success(
                "Saved control changes. They will apply at the next cycle boundary."
            )

    st.subheader("Evaluator Training Diagnostics")
    _render_evaluator_training_diagnostics_section(st=st, work_dir=paths.work_dir)

    st.subheader("Plots")
    st.caption("Time-series plots use absolute UTC timestamps from bootstrap history.")
    downsampled_tree_num_nodes = _downsample_series(dashboard_data.tree_num_nodes)
    downsampled_dataset_num_rows = _downsample_series(dashboard_data.dataset_num_rows)
    downsampled_active_evaluator = _downsample_series(dashboard_data.active_evaluator)
    downsampled_evaluator_losses = _downsample_loss_series_by_name(
        dashboard_data.evaluator_loss_by_name
    )
    downsampled_certified_record_score = _downsample_series(
        dashboard_data.certified_record_score
    )
    plot_columns = st.columns(2)
    with plot_columns[0]:
        _render_plot(st, lambda: plot_tree_size(downsampled_tree_num_nodes))
        _render_plot(st, lambda: plot_dataset_size(downsampled_dataset_num_rows))
    with plot_columns[1]:
        _render_plot(st, lambda: plot_active_evaluator(downsampled_active_evaluator))
        loss_log_scale = (
            st.toggle("Log scale for loss", value=False)
            if hasattr(st, "toggle")
            else st.checkbox("Log scale for loss", value=False)
        )
        _render_plot(
            st,
            lambda: plot_evaluator_losses(
                downsampled_evaluator_losses,
                log_scale=loss_log_scale,
            ),
        )
        if not loss_series_contains_points(dashboard_data.evaluator_loss_by_name):
            st.caption(
                "No evaluator loss data found yet. Checked training_status.json files: "
                + checked_training_status_files_summary(paths)
            )

    st.subheader("Certified Record Progress")
    st.caption(
        "Artifact-pipeline record progress is sourced from "
        "pipeline/generation_*/dataset_status.json when available."
    )
    if _has_known_optional_series_values(downsampled_certified_record_score):
        _render_plot(
            st,
            lambda: plot_certified_record_score(downsampled_certified_record_score),
        )
    else:
        st.caption("No certified record yet.")

    st.subheader("Tree Structure")
    _render_tree_structure_section(
        st=st,
        tree_status=dashboard_data.latest_tree_status,
        depth_distribution=dashboard_data.latest_tree_depth_distribution,
    )

    st.subheader("Tree / State Inspector")
    _render_tree_inspector_fragment(
        st=st,
        paths=paths,
        latest_linoo_selection_table=dashboard_data.latest_linoo_selection_table,
        tree_node_classification_summary=(
            dashboard_data.latest_tree_node_classification_summary
        ),
    )

    st.subheader("Debug Info")
    st.write(
        "Last checkpoint path:",
        _format_value(
            run_state.latest_runtime_checkpoint_path
            or run_state.metadata.get("runtime_checkpoint_path")
        ),
    )
    st.write(
        "Effective runtime:",
        run_state.metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY, {}),
    )
    st.write(
        "Effective runtime hash:",
        _format_value(
            run_state.metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY)
        ),
    )
    st.write("Applied control:")
    st.json(bootstrap_control_to_dict(applied_control))
    st.write("Current control file:")
    st.json(bootstrap_control_to_dict(control))


def _get_streamlit() -> Any:
    """Return the Streamlit module or raise a helpful runtime error."""
    try:
        return import_module("streamlit")
    except ModuleNotFoundError as exc:
        raise MissingStreamlitDashboardDependencyError from exc


class MissingStreamlitDashboardDependencyError(RuntimeError):
    """Raised when the local dashboard is requested without Streamlit installed."""

    def __init__(self) -> None:
        """Initialize the missing-Streamlit dependency error."""
        super().__init__(
            "Streamlit is not installed. Install `streamlit` to use the local dashboard."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    cli_args = parser.parse_args()
    run_dashboard_app(cli_args.work_dir)


__all__ = ["run_dashboard_app"]
