"""Local Streamlit dashboard for monitoring and controlling Morpion bootstrap runs."""

from __future__ import annotations

import argparse
import time
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.bootstrap_loop import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.control import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    bootstrap_control_to_dict,
    load_bootstrap_control,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
    _checked_training_status_files_summary,
    _loss_series_contains_points,
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
    format_bool_icon as _format_bool_icon,
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
    plot_tree_depth_distribution,
    plot_tree_size,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.disk_usage import (
    render_disk_usage_section as _render_disk_usage_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    _observability_summary_from_metadata,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
    render_observability_section as _render_observability_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
    render_plot as _render_plot,
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
from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
    build_morpion_bootstrap_tree_inspector_snapshot,
)
from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
    MorpionEvaluatorDiagnosticExample,
    MorpionEvaluatorTrainingDiagnostics,
    load_latest_evaluator_training_diagnostics,
)
from chipiron.environments.morpion.bootstrap.streamlit_morpion_clickable_board import (
    render_clickable_morpion_board,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        MorpionBootstrapCertifiedRecordBoardView,
        TreeDepthDistributionRow,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
        MorpionBootstrapChildSummary,
    )
    from chipiron.environments.morpion.bootstrap.history import (
        MorpionBootstrapTreeStatus,
    )

TREE_INSPECTOR_TIMING_PREFIX = "[tree-inspector-timing]"


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
        configured_evaluator_names=configured_evaluator_names,
        current_force_evaluator=control.force_evaluator,
    )
    baseline_tree_branch_limit = _baseline_tree_branch_limit(config)
    applied_runtime_control = _applied_runtime_control(run_state)
    effective_runtime_config = _effective_runtime_config(run_state)
    effective_runtime_hash = _effective_runtime_hash(run_state)
    tree_branch_limit_input_value = _tree_branch_limit_input_value(
        runtime_control=control.runtime,
        baseline_tree_branch_limit=baseline_tree_branch_limit,
    )
    pending_fields = _pending_control_fields(control, applied_control)
    pending_sections = _pending_control_sections(control, applied_control)
    dataset_summary = _dataset_status_summary(config, control, applied_control)
    scheduling_summary = _scheduling_status_summary(config, control, applied_control)
    evaluator_summary = _evaluator_control_status_summary(
        control=control,
        applied_control=applied_control,
        configured_evaluator_names=configured_evaluator_names,
    )
    runtime_summary = _runtime_status_summary(
        baseline_tree_branch_limit=baseline_tree_branch_limit,
        current_runtime_control=control.runtime,
        applied_runtime_control=applied_runtime_control,
        effective_runtime_config=effective_runtime_config,
        effective_runtime_hash=effective_runtime_hash,
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
        summary=_observability_summary_from_metadata(run_state.metadata),
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
            baseline_tree_branch_limit=baseline_tree_branch_limit,
            effective_runtime_config=effective_runtime_config,
            latest_dataset_rows=latest_dataset_rows,
            pending_changes=pending_changes,
            configured_evaluator_names=configured_evaluator_names,
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
                configured_evaluator_names=configured_evaluator_names,
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
        if not _loss_series_contains_points(dashboard_data.evaluator_loss_by_name):
            st.caption(
                "No evaluator loss data found yet. Checked training_status.json files: "
                + _checked_training_status_files_summary(paths)
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


def _tree_inspector_rerun(st: Any) -> None:
    """Rerun only the tree-inspector fragment when supported."""
    try:
        st.rerun(scope="fragment")
    except TypeError:
        st.rerun()


def _render_tree_inspector_section(
    *,
    st: Any,
    paths: MorpionBootstrapPaths,
    latest_linoo_selection_table: Any,
    tree_node_classification_summary: Any,
) -> None:
    """Render the bounded runtime-tree inspector for the latest checkpoint."""
    section_start_time = time.perf_counter()
    state_key = f"morpion_bootstrap_selected_node::{paths.work_dir}"
    selected_node_id = st.session_state.get(state_key)

    _render_linoo_selection_table(
        st=st,
        latest_linoo_selection_table=latest_linoo_selection_table,
    )
    _render_tree_node_classification_summary(
        st=st,
        summary=tree_node_classification_summary,
    )

    snapshot_start_time = time.perf_counter()
    snapshot = build_morpion_bootstrap_tree_inspector_snapshot(
        paths.work_dir,
        selected_node_id=selected_node_id,
    )
    snapshot_duration = time.perf_counter() - snapshot_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} build_snapshot "
        f"work_dir={paths.work_dir.name} selected_node_id={selected_node_id!r} "
        f"checkpoint={None if snapshot.checkpoint_path is None else snapshot.checkpoint_path.name} "
        f"total_s={snapshot_duration:.6f}",
        flush=True,
    )

    if snapshot.status_message is not None:
        st.info(snapshot.status_message)
    if snapshot.error_message is not None:
        st.warning(snapshot.error_message)
        return
    if snapshot.selected_node_id is None or snapshot.node_summary is None:
        st.caption("No persisted runtime checkpoint available yet.")
        return
    if snapshot.selection_warning is not None:
        st.warning(snapshot.selection_warning)

    st.session_state[state_key] = snapshot.selected_node_id
    _render_tree_inspector_navigation(
        st=st,
        snapshot=snapshot,
        state_key=state_key,
    )

    node_summary = snapshot.node_summary
    summary_columns = st.columns(4)
    summary_columns[0].metric("Selected Node", node_summary.node_id)
    summary_columns[1].metric("Depth", _format_value(node_summary.depth))
    summary_columns[2].metric("Children", str(node_summary.num_children))
    summary_columns[3].metric(
        "Best Branch",
        _format_value(node_summary.best_branch_label),
    )

    details_columns = st.columns(2)
    with details_columns[0]:
        with st.expander("Node Summary"):
            st.json(_tree_inspector_node_summary_dict(snapshot))
        with st.expander("Local Tree"):
            st.json(_tree_inspector_local_tree_dict(snapshot))
    with details_columns[1]:
        if snapshot.state_view is not None:
            st.caption("Selected Morpion State")
            clickable_board_start_time = time.perf_counter()
            board_click_event = render_clickable_morpion_board(
                svg=snapshot.state_view.board_svg,
                click_targets=snapshot.state_view.board_click_targets,
                click_radius=snapshot.state_view.board_click_radius,
                height=760,
                render_size=snapshot.state_view.board_render_size,
                key=f"{state_key}::board::{snapshot.selected_node_id}",
            )
            clickable_board_duration = time.perf_counter() - clickable_board_start_time
            print(
                f"{TREE_INSPECTOR_TIMING_PREFIX} render_clickable_board "
                f"work_dir={paths.work_dir.name} selected_node_id={snapshot.selected_node_id!r} "
                f"total_s={clickable_board_duration:.6f}",
                flush=True,
            )
            if board_click_event is not None:
                click_nonce = board_click_event.get("click_nonce")
                click_nonce_key = f"{state_key}::board_click_nonce"
                if click_nonce != st.session_state.get(click_nonce_key):
                    st.session_state[click_nonce_key] = click_nonce
                    clicked_action_name = board_click_event.get("action_name")
                    if isinstance(clicked_action_name, str):
                        selected_child_node_id = _selected_child_node_id_for_branch(
                            snapshot.child_summaries,
                            clicked_action_name,
                        )
                        if selected_child_node_id is not None:
                            st.session_state[state_key] = selected_child_node_id
                            _tree_inspector_rerun(st)
                        else:
                            st.caption(
                                f"Action {_format_value(clicked_action_name)} is not expanded in this checkpoint."
                            )
            with st.expander("ASCII Board"):
                st.code(snapshot.state_view.board_text)

    st.caption("Outgoing actions")
    _render_tree_inspector_outgoing_actions(
        st=st,
        snapshot=snapshot,
        state_key=state_key,
    )
    section_duration = time.perf_counter() - section_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} render_tree_inspector_section "
        f"work_dir={paths.work_dir.name} selected_node_id={snapshot.selected_node_id!r} "
        f"total_s={section_duration:.6f}",
        flush=True,
    )


def _render_tree_inspector_fragment(
    *,
    st: Any,
    paths: MorpionBootstrapPaths,
    latest_linoo_selection_table: Any,
    tree_node_classification_summary: Any,
) -> None:
    """Render the tree inspector in a fragment when supported by Streamlit."""
    fragment_renderer = (
        st.fragment(_render_tree_inspector_section)
        if hasattr(st, "fragment")
        else _render_tree_inspector_section
    )
    fragment_renderer(
        st=st,
        paths=paths,
        latest_linoo_selection_table=latest_linoo_selection_table,
        tree_node_classification_summary=tree_node_classification_summary,
    )


def _render_linoo_selection_table(
    *,
    st: Any,
    latest_linoo_selection_table: Any,
) -> None:
    """Render the latest persisted Linoo depth-selection table."""
    st.markdown("**Latest Linoo depth selection table**")
    st.caption(
        "Linoo selects the depth with minimal opened_count * (depth + 1), "
        "tie-breaking by smaller depth."
    )
    rows = _linoo_selection_table_rows(latest_linoo_selection_table)
    if not rows:
        st.caption("No Linoo selection table available yet.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)


def _linoo_selection_table_rows(
    latest_linoo_selection_table: Any,
) -> list[dict[str, object]]:
    """Return dashboard rows for the latest Linoo table artifact."""
    rows = getattr(latest_linoo_selection_table, "rows", None)
    if rows is None:
        return []
    return [
        {
            "depth": row.depth,
            "opened_count": row.opened,
            "frontier_count": row.frontier,
            "deterministic_index": row.deterministic_index,
            "weight": row.weight,
            "probability": row.probability,
            "best_node_id": row.best_node,
            "best_direct_value": row.best_value,
            "selected": row.selected,
        }
        for row in rows
    ]


def _format_tree_node_classification_metric(
    count: int,
    *,
    total_nodes: int,
    percentages_available: bool,
) -> str:
    """Render one count and, when safe, its total-tree percentage."""
    if not percentages_available or total_nodes <= 0:
        return str(count)
    return f"{count} ({(count / total_nodes) * 100.0:.1f}%)"


def _render_tree_node_classification_summary(
    *,
    st: Any,
    summary: Any,
) -> None:
    """Render compact exact and terminal node proportions for the latest tree."""
    if summary is None:
        st.caption("No latest tree snapshot classification summary available yet.")
        return

    unknown_nodes = getattr(summary, "unknown_classification_nodes", 0)
    total_nodes = getattr(summary, "total_nodes", 0)
    percentages_available = unknown_nodes == 0
    summary_columns = st.columns(5)
    summary_columns[0].metric("Total Nodes", str(total_nodes))
    summary_columns[1].metric(
        "Exact Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "exact_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[2].metric(
        "Terminal Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[3].metric(
        "Exact Terminal Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "exact_terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[4].metric(
        "Non-Exact Non-Terminal",
        _format_tree_node_classification_metric(
            getattr(summary, "non_exact_non_terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    if unknown_nodes > 0:
        st.caption(
            "Some snapshot nodes lack exact/terminal flags; percentages are omitted."
        )


def _render_tree_structure_section(
    *,
    st: Any,
    tree_status: MorpionBootstrapTreeStatus | None,
    depth_distribution: tuple[TreeDepthDistributionRow, ...],
) -> None:
    """Render one compact tree-structure summary with per-depth counts."""
    if tree_status is None:
        st.caption("No tree structure has been recorded yet.")
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("Total Nodes", str(tree_status.num_nodes))
    summary_columns[1].metric(
        "Expanded Nodes",
        _format_value(tree_status.num_expanded_nodes),
    )
    summary_columns[2].metric("Min Depth", _format_value(tree_status.min_depth_present))
    summary_columns[3].metric("Max Depth", _format_value(tree_status.max_depth_present))

    if not depth_distribution:
        st.caption("No tree depth distribution available yet.")
        return
    _render_plot(st, lambda: plot_tree_depth_distribution(depth_distribution))
    rows = _tree_structure_rows(depth_distribution)
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_record_status_section(
    *,
    st: Any,
    certified_status: Any,
    frontier_status: Any,
) -> None:
    """Render a strict certified record summary alongside the frontier best."""
    certified_columns = st.columns(4)
    if certified_status is None or certified_status.current_best_total_points is None:
        certified_columns[0].metric(
            "Certified record total points", "No certified record yet"
        )
        certified_columns[1].metric("Certified record moves", "n/a")
        certified_columns[2].metric("Certified exact", "n/a")
        certified_columns[3].metric("Certified terminal", "n/a")
    else:
        certified_columns[0].metric(
            "Certified record total points",
            _format_value(certified_status.current_best_total_points),
        )
        certified_columns[1].metric(
            "Certified record moves",
            _format_value(certified_status.current_best_moves_since_start),
        )
        certified_columns[2].metric(
            "Certified exact",
            _format_value(certified_status.current_best_is_exact),
        )
        certified_columns[3].metric(
            "Certified terminal",
            _format_value(certified_status.current_best_is_terminal),
        )

    frontier_columns = st.columns(4)
    frontier_columns[0].metric(
        "Frontier best total points",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_total_points
        ),
    )
    frontier_columns[1].metric(
        "Frontier best moves",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_moves_since_start
        ),
    )
    frontier_columns[2].metric(
        "Frontier exact",
        _format_value(
            None if frontier_status is None else frontier_status.current_best_is_exact
        ),
    )
    frontier_columns[3].metric(
        "Frontier terminal",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_is_terminal
        ),
    )
    st.caption(
        "Frontier best source: "
        + _format_value(
            None if frontier_status is None else frontier_status.current_best_source
        )
    )


def _render_evaluator_training_diagnostics_section(*, st: Any, work_dir: Path) -> None:
    """Render the latest persisted evaluator diagnostics for one work directory."""
    diagnostics_by_evaluator = (
        _load_latest_evaluator_training_diagnostics_for_dashboard(work_dir)
    )
    if not diagnostics_by_evaluator:
        st.caption("No evaluator diagnostics have been saved yet.")
        return

    evaluator_names = tuple(sorted(diagnostics_by_evaluator))
    selected_evaluator_name = st.selectbox(
        "Diagnostics evaluator",
        options=evaluator_names,
        key="evaluator_training_diagnostics_name",
    )
    diagnostics = diagnostics_by_evaluator[selected_evaluator_name]
    summary_columns = st.columns(5)
    summary_columns[0].metric("Generation", diagnostics.generation)
    summary_columns[1].metric("Dataset Size", diagnostics.dataset_size)
    summary_columns[2].metric("MAE Before", _format_value(diagnostics.mae_before))
    summary_columns[3].metric("MAE After", _format_value(diagnostics.mae_after))
    summary_columns[4].metric(
        "Max Error After",
        _format_value(diagnostics.max_abs_error_after),
    )
    st.caption(
        f"Created at {diagnostics.created_at} UTC. "
        "Representative rows are deterministic scale windows; worst rows are sorted by post-training absolute error."
    )
    st.markdown("Representative Examples")
    st.dataframe(
        _diagnostic_examples_rows(diagnostics.representative_examples),
        width="stretch",
        hide_index=True,
    )
    st.markdown("Worst Error Examples")
    worst_rows = _diagnostic_examples_rows(diagnostics.worst_examples)
    if worst_rows:
        st.dataframe(worst_rows, width="stretch", hide_index=True)
    else:
        st.caption(
            "No post-training predictions were available for worst-error ranking."
        )


def _render_current_certified_record_board_section(
    *,
    st: Any,
    board_view: MorpionBootstrapCertifiedRecordBoardView | None,
) -> None:
    """Render the current strict certified Morpion record board when available."""
    if board_view is None:
        st.caption("No certified record state available yet.")
        return

    summary_columns = st.columns(5)
    summary_columns[0].metric("Total Points", str(board_view.total_points))
    summary_columns[1].metric("Moves Since Start", str(board_view.moves_since_start))
    summary_columns[2].metric("Exact", _format_value(board_view.is_exact))
    summary_columns[3].metric("Terminal", _format_value(board_view.is_terminal))
    summary_columns[4].metric("Source", board_view.source)
    st.components.v1.html(board_view.board_svg, height=760)
    if board_view.board_text is not None:
        st.code(board_view.board_text)


def _tree_structure_rows(
    depth_distribution: tuple[TreeDepthDistributionRow, ...],
) -> list[dict[str, int]]:
    """Return one dashboard-friendly per-depth node-count table."""
    return [
        {
            "depth": row.depth,
            "num_nodes": row.num_nodes,
            "cumulative_nodes": row.cumulative_nodes,
        }
        for row in depth_distribution
    ]


def _load_latest_evaluator_training_diagnostics_for_dashboard(
    work_dir: str | Path,
) -> dict[str, MorpionEvaluatorTrainingDiagnostics]:
    """Load latest evaluator diagnostics, tolerating absent artifacts."""
    return load_latest_evaluator_training_diagnostics(work_dir)


def _diagnostic_examples_rows(
    examples: Sequence[MorpionEvaluatorDiagnosticExample],
) -> list[dict[str, object | None]]:
    """Return dashboard-friendly diagnostic example rows."""
    return [
        {
            "row_index": example.row_index,
            "node_id": example.node_id,
            "state_tag": example.state_tag,
            "depth": example.depth,
            "target_value": example.target_value,
            "prediction_before": example.prediction_before,
            "prediction_after": example.prediction_after,
            "abs_error_before": example.abs_error_before,
            "abs_error_after": example.abs_error_after,
        }
        for example in examples
    ]


def _has_known_optional_series_values(series: tuple[Any, ...]) -> bool:
    """Return whether one optional-value time series contains any known value."""
    return any(getattr(point, "value", None) is not None for point in series)


def _render_tree_inspector_navigation(
    *,
    st: Any,
    snapshot: Any,
    state_key: str,
) -> None:
    """Render root/parent/child/direct node navigation controls."""
    local_tree_view = snapshot.local_tree_view
    node_summary = snapshot.node_summary
    if local_tree_view is None or node_summary is None:
        return

    nav_columns = st.columns((1, 1, 2, 3))
    if nav_columns[0].button("Go to root", key=f"{state_key}::root"):
        st.session_state[state_key] = local_tree_view.root_node_id
        _tree_inspector_rerun(st)
    parent_disabled = not node_summary.parent_ids
    if nav_columns[1].button(
        "Go to parent",
        key=f"{state_key}::parent",
        disabled=parent_disabled,
    ):
        st.session_state[state_key] = node_summary.parent_ids[0]
        _tree_inspector_rerun(st)

    child_options = [summary.branch_label for summary in snapshot.child_summaries]
    selected_branch = nav_columns[2].selectbox(
        "Child branch",
        options=child_options if child_options else [""],
        key=f"{state_key}::child_branch",
        disabled=not child_options,
        label_visibility="collapsed",
    )
    if nav_columns[3].button(
        "Go to child",
        key=f"{state_key}::child",
        disabled=not child_options,
    ):
        selected_child_node_id = _selected_child_node_id_for_branch(
            snapshot.child_summaries,
            selected_branch,
        )
        if selected_child_node_id is not None:
            st.session_state[state_key] = selected_child_node_id
            _tree_inspector_rerun(st)

    selected_node_input = st.text_input(
        "Node id",
        value=snapshot.selected_node_id,
        key=f"{state_key}::node_input",
        help="Jump directly to a checkpoint node id.",
    )
    if st.button("Go to node id", key=f"{state_key}::node_jump"):
        st.session_state[state_key] = selected_node_input.strip()
        _tree_inspector_rerun(st)


def _selected_child_node_id_for_branch(
    child_summaries: tuple[MorpionBootstrapChildSummary, ...],
    branch_label: str,
) -> str | None:
    """Return the expanded child node id for the selected branch row."""
    for child_summary in child_summaries:
        if child_summary.branch_label == branch_label:
            return child_summary.child_node_id
    return None


def _tree_inspector_node_summary_dict(snapshot: Any) -> dict[str, object]:
    """Return one JSON-friendly selected-node summary for the dashboard."""
    node_summary = snapshot.node_summary
    if node_summary is None:
        return {}
    return {
        "node_id": node_summary.node_id,
        "depth": node_summary.depth,
        "parent_ids": list(node_summary.parent_ids),
        "child_ids": list(node_summary.child_ids),
        "visit_count": node_summary.visit_count,
        "is_terminal": node_summary.is_terminal,
        "is_exact": node_summary.is_exact,
        "direct_value_scalar": node_summary.direct_value_scalar,
        "backed_up_value_scalar": node_summary.backed_up_value_scalar,
        "best_child_id": node_summary.best_child_id,
        "best_branch_label": node_summary.best_branch_label,
    }


def _tree_inspector_local_tree_dict(snapshot: Any) -> dict[str, object]:
    """Return one JSON-friendly bounded tree neighborhood summary."""
    local_tree_view = snapshot.local_tree_view
    if local_tree_view is None:
        return {}
    return {
        "root_node_id": local_tree_view.root_node_id,
        "selected_node_id": local_tree_view.selected_node_id,
        "parent_node_ids": list(local_tree_view.parent_node_ids),
        "sibling_node_ids": list(local_tree_view.sibling_node_ids),
        "child_node_ids": list(local_tree_view.child_node_ids),
    }


def _tree_inspector_child_rows(snapshot: Any) -> list[dict[str, object]]:
    """Return the child/action rows shown in the inspector table."""
    return [
        {
            "branch": child_summary.branch_label,
            "child_node_id": child_summary.child_node_id,
            "display_value": child_summary.display_value_scalar,
            "backed_up_value": child_summary.backed_up_value_scalar,
            "direct_value": child_summary.direct_value_scalar,
            "visit_count": child_summary.visit_count,
            "is_exact": _format_bool_icon(child_summary.is_exact),
            "is_terminal": _format_bool_icon(child_summary.is_terminal),
        }
        for child_summary in snapshot.child_summaries
    ]


def _render_tree_inspector_outgoing_actions(
    *,
    st: Any,
    snapshot: Any,
    state_key: str,
) -> None:
    """Render one compact outgoing-actions list with direct child navigation."""
    child_rows = _tree_inspector_child_rows(snapshot)
    if not child_rows:
        st.caption("No outgoing actions available.")
        return

    row_columns = st.columns((4, 2, 2, 2, 2, 1, 1, 1))
    row_columns[0].caption("Branch")
    row_columns[1].caption("Child Node")
    row_columns[2].caption("Display")
    row_columns[3].caption("Direct")
    row_columns[4].caption("Backed-up")
    row_columns[5].caption("Exact")
    row_columns[6].caption("Terminal")
    row_columns[7].caption("Go")

    for index, child_row in enumerate(child_rows):
        child_node_id = child_row["child_node_id"]
        branch = child_row["branch"]
        row_columns = st.columns((4, 2, 2, 2, 2, 1, 1, 1))
        row_columns[0].write(_format_value(branch))
        row_columns[1].write(_format_value(child_node_id))
        row_columns[2].write(_format_value(child_row["display_value"]))
        row_columns[3].write(_format_value(child_row["direct_value"]))
        row_columns[4].write(_format_value(child_row["backed_up_value"]))
        row_columns[5].write(_format_value(child_row["is_exact"]))
        row_columns[6].write(_format_value(child_row["is_terminal"]))
        if row_columns[7].button(
            "Go",
            key=(
                f"{state_key}::child_row::{index}::"
                f"{_format_value(branch)}::{_format_value(child_node_id)}"
            ),
            disabled=child_node_id is None,
        ):
            st.session_state[state_key] = child_node_id
            _tree_inspector_rerun(st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    cli_args = parser.parse_args()
    run_dashboard_app(cli_args.work_dir)


__all__ = ["run_dashboard_app"]
