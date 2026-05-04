"""Local Streamlit dashboard for monitoring and controlling Morpion bootstrap runs."""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .history import MorpionBootstrapTreeStatus

from .bootstrap_loop import MorpionBootstrapPaths
from .config import (
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapConfig,
    load_bootstrap_config,
)
from .control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapRuntimeControl,
    bootstrap_control_from_metadata,
    bootstrap_control_to_dict,
    bootstrap_runtime_control_from_metadata,
    effective_runtime_config_from_metadata,
    load_bootstrap_control,
    save_bootstrap_control,
)
from .dashboard_plot import (
    plot_active_evaluator,
    plot_certified_record_score,
    plot_dataset_size,
    plot_evaluator_losses,
    plot_tree_depth_distribution,
    plot_tree_size,
)
from .evaluator_diagnostics import (
    MorpionEvaluatorDiagnosticExample,
    MorpionEvaluatorTrainingDiagnostics,
    load_latest_evaluator_training_diagnostics,
)
from .evaluator_family import canonical_morpion_evaluator_names
from .history_view import (
    DiskUsageSummary,
    MorpionBootstrapCertifiedRecordBoardView,
    TreeDepthDistributionRow,
    build_current_certified_record_board_view,
    build_morpion_bootstrap_dashboard_data,
    format_num_bytes,
)
from .process_control import (
    MorpionBootstrapProcessControlError,
    MorpionBootstrapProcessState,
    launcher_command_for_work_dir,
    load_morpion_bootstrap_process_state,
    restart_morpion_bootstrap_process,
    start_morpion_bootstrap_process,
    stop_morpion_bootstrap_process,
)
from .run_state import initialize_bootstrap_run_state, load_bootstrap_run_state
from .streamlit_morpion_clickable_board import render_clickable_morpion_board
from .tree_inspector import build_morpion_bootstrap_tree_inspector_snapshot

MAX_PLOT_POINTS = 2000
TREE_INSPECTOR_TIMING_PREFIX = "[tree-inspector-timing]"


def _path_mtime_ns(path: Path) -> int:
    """Return one path freshness token, treating missing paths as zero."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _loss_series_contains_points(
    loss_by_name: Mapping[str, tuple[Any, ...]],
) -> bool:
    """Return whether any evaluator loss series contains at least one concrete point."""
    return any(series for series in loss_by_name.values())


def _checked_training_status_files_summary(paths: MorpionBootstrapPaths) -> str:
    """Summarize the training-status files the dashboard inspected for loss data."""
    checked_files = sorted(paths.pipeline_dir.glob("generation_*/training_status.json"))
    if not checked_files:
        return "none"
    rendered = [paths.relative_to_work_dir(path) for path in checked_files[-5:]]
    omitted_count = len(checked_files) - len(rendered)
    if omitted_count > 0:
        return ", ".join(rendered) + f" (+{omitted_count} more)"
    return ", ".join(rendered)


def _latest_generation_json_path(directory: Path) -> Path | None:
    """Return the newest generation JSON file in one directory when present."""
    candidates = sorted(directory.glob("generation_*.json"))
    return None if not candidates else candidates[-1]


def _cached_dashboard_data_freshness_tokens(
    paths: MorpionBootstrapPaths,
) -> tuple[int, ...]:
    """Return freshness tokens for dashboard-wide data rebuilds."""
    latest_tree_snapshot_path = _latest_generation_json_path(paths.tree_snapshot_dir)
    latest_runtime_checkpoint_path = _latest_generation_json_path(
        paths.runtime_checkpoint_dir
    )
    return (
        _path_mtime_ns(paths.work_dir),
        _path_mtime_ns(paths.bootstrap_config_path),
        _path_mtime_ns(paths.control_path),
        _path_mtime_ns(paths.run_state_path),
        _path_mtime_ns(paths.history_jsonl_path),
        _path_mtime_ns(paths.latest_status_path),
        _path_mtime_ns(paths.launcher_pid_path),
        _path_mtime_ns(paths.launcher_process_state_path),
        _path_mtime_ns(paths.launcher_stdout_log_path),
        _path_mtime_ns(paths.launcher_stderr_log_path),
        _path_mtime_ns(paths.tree_snapshot_dir),
        _path_mtime_ns(paths.runtime_checkpoint_dir),
        _path_mtime_ns(paths.rows_dir),
        _path_mtime_ns(paths.model_dir),
        _path_mtime_ns(paths.latest_linoo_selection_table_path),
        0
        if latest_tree_snapshot_path is None
        else _path_mtime_ns(latest_tree_snapshot_path),
        (
            0
            if latest_runtime_checkpoint_path is None
            else _path_mtime_ns(latest_runtime_checkpoint_path)
        ),
    )


def _cached_certified_record_board_freshness_tokens(
    paths: MorpionBootstrapPaths,
) -> tuple[int, ...]:
    """Return freshness tokens for certified-record board rebuilds."""
    latest_tree_snapshot_path = _latest_generation_json_path(paths.tree_snapshot_dir)
    return (
        _path_mtime_ns(paths.run_state_path),
        _path_mtime_ns(paths.history_jsonl_path),
        _path_mtime_ns(paths.tree_snapshot_dir),
        0
        if latest_tree_snapshot_path is None
        else _path_mtime_ns(latest_tree_snapshot_path),
    )


@lru_cache(maxsize=1)
def _cached_build_morpion_bootstrap_dashboard_data(
    work_dir: str,
    freshness_tokens: tuple[int, ...],
) -> Any:
    """Cache dashboard-wide data for one work dir until relevant artifacts change."""
    _ = freshness_tokens
    return build_morpion_bootstrap_dashboard_data(work_dir)


@lru_cache(maxsize=1)
def _cached_build_current_certified_record_board_view(
    work_dir: str,
    freshness_tokens: tuple[int, ...],
) -> MorpionBootstrapCertifiedRecordBoardView | None:
    """Cache the certified-record board view until its source artifacts change."""
    _ = freshness_tokens
    return build_current_certified_record_board_view(work_dir)


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
        force_evaluator = st.selectbox(
            "Forced evaluator",
            options=selectable_force_evaluator_options,
            index=_force_evaluator_option_index(
                selectable_force_evaluator_options,
                control.force_evaluator,
            ),
            disabled=(force_evaluator_mode != "forced" or not force_evaluator_options),
            format_func=lambda value: _format_force_evaluator_option(
                value,
                configured_evaluator_names=configured_evaluator_names,
            ),
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


class InvalidDashboardPlotPointLimitError(ValueError):
    """Raised when dashboard plot downsampling is configured with an invalid cap."""

    def __init__(self, max_points: int) -> None:
        """Initialize the invalid plot-point-limit error."""
        super().__init__(
            f"Dashboard plot max_points must be at least 1, got {max_points}."
        )


def _render_plot(st: Any, build_plot: Any) -> None:
    """Render one existing matplotlib plot helper into Streamlit."""
    build_plot()
    figure = plt.gcf()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def _render_disk_usage_section(
    *,
    st: Any,
    summary: DiskUsageSummary,
) -> None:
    """Render operator-facing run and device disk usage information."""
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Run Dir Size",
        format_num_bytes(summary.run_dir_num_bytes),
        delta=_format_disk_usage_pct(summary.run_dir_pct_of_device_total),
    )
    metric_columns[1].metric(
        "Device Free Space",
        format_num_bytes(summary.device_free_num_bytes),
    )
    metric_columns[2].metric(
        "Device Used Space",
        format_num_bytes(summary.device_used_num_bytes),
    )
    metric_columns[3].metric(
        "Device Total Space",
        format_num_bytes(summary.device_total_num_bytes),
    )
    breakdown_rows = [
        {"artifact_group": row.label, "size": format_num_bytes(row.num_bytes)}
        for row in summary.breakdown_rows
    ]
    st.dataframe(breakdown_rows, width="stretch", hide_index=True)
    st.caption(
        "Run breakdown is sorted largest-first. Retention keeps only the latest "
        "checkpoint and tree export by default."
    )


def _format_disk_usage_pct(value: float | None) -> str:
    """Format one optional disk-usage percentage for dashboard metrics."""
    if value is None:
        return "unknown"
    return f"{value:.2f}% of device"


def _downsample_series[SeriesPointT](
    series: Sequence[SeriesPointT],
    max_points: int = MAX_PLOT_POINTS,
) -> tuple[SeriesPointT, ...]:
    """Return one bounded series while preserving the first and last points."""
    if max_points < 1:
        raise InvalidDashboardPlotPointLimitError(max_points)
    if len(series) <= max_points:
        return tuple(series)
    if max_points == 1:
        return (series[-1],)
    last_index = len(series) - 1
    sampled_indices = tuple(
        int(sample_index * last_index / (max_points - 1))
        for sample_index in range(max_points)
    )
    return tuple(series[index] for index in sampled_indices)


def _downsample_loss_series_by_name[SeriesPointT](
    loss_by_name: Mapping[str, Sequence[SeriesPointT]],
    max_points: int = MAX_PLOT_POINTS,
) -> dict[str, tuple[SeriesPointT, ...]]:
    """Return one bounded evaluator-loss mapping keyed by evaluator name."""
    return {
        evaluator_name: _downsample_series(series, max_points=max_points)
        for evaluator_name, series in loss_by_name.items()
    }


def _render_run_control_section(*, st: Any, paths: MorpionBootstrapPaths) -> None:
    """Render Start / Stop / Restart controls for the launcher subprocess."""
    st.subheader("Run Control")
    state = load_morpion_bootstrap_process_state(paths)
    _render_run_control_state(st=st, paths=paths, state=state)

    button_columns = st.columns(3)
    try:
        if button_columns[0].button("Start", disabled=state.is_running):
            result = start_morpion_bootstrap_process(paths)
            if result.already_running:
                st.warning("Launcher is already running.")
            else:
                st.success("Launcher started.")
            st.rerun()
        if button_columns[1].button("Stop", disabled=not state.is_running):
            stop_morpion_bootstrap_process(paths)
            st.success("Launcher stopped.")
            st.rerun()
        if button_columns[2].button("Restart"):
            restart_morpion_bootstrap_process(paths)
            st.success("Launcher restarted.")
            st.rerun()
    except MorpionBootstrapProcessControlError as exc:
        st.error(str(exc))


def _render_run_control_state(
    *,
    st: Any,
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapProcessState,
) -> None:
    """Render one compact launcher-process status panel."""
    columns = st.columns(4)
    columns[0].metric("Launcher status", state.status_label)
    columns[1].metric("PID", _format_value(state.pid))
    columns[2].metric("Started at", _format_value(state.started_at_utc))
    columns[3].metric("Last stop reason", _format_value(state.last_stop_reason))
    st.code(
        _render_launcher_command_text(
            state.command or launcher_command_for_work_dir(paths.work_dir)
        )
    )
    st.write("stdout log:", str(paths.launcher_stdout_log_path))
    st.write("stderr log:", str(paths.launcher_stderr_log_path))


def _render_launcher_command_text(command: tuple[str, ...]) -> str:
    """Render one launcher command tuple as a stable shell-style string."""
    return " ".join(command)


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
            "selection_index": row.index,
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
    child_summaries: tuple[Any, ...],
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


def _load_run_state(paths: MorpionBootstrapPaths) -> Any:
    """Return the latest run state when present, else one initialized default state."""
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path)
    return initialize_bootstrap_run_state()


def _load_applied_control(paths: MorpionBootstrapPaths) -> MorpionBootstrapControl:
    """Return the last control known to have been applied at a cycle boundary."""
    if not paths.run_state_path.is_file():
        return MorpionBootstrapControl()
    run_state = load_bootstrap_run_state(paths.run_state_path)
    return bootstrap_control_from_metadata(
        run_state.metadata.get(BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY)
    )


def _load_bootstrap_config_or_none(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapConfig | None:
    """Return the persisted bootstrap config when present."""
    if not paths.bootstrap_config_path.is_file():
        return None
    return load_bootstrap_config(paths.bootstrap_config_path)


def _baseline_tree_branch_limit(config: MorpionBootstrapConfig | None) -> int:
    """Return the configured baseline tree branch limit or the stable default."""
    if config is None:
        return DEFAULT_MORPION_TREE_BRANCH_LIMIT
    return config.runtime.tree_branch_limit


def _applied_runtime_control(run_state: Any) -> MorpionBootstrapRuntimeControl:
    """Return the last applied runtime-control subsection from run-state metadata."""
    return bootstrap_runtime_control_from_metadata(
        getattr(run_state, "metadata", {}).get(
            BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY
        )
    )


def _effective_runtime_config(
    run_state: Any,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Return the effective runtime config from run-state metadata when present."""
    return effective_runtime_config_from_metadata(
        getattr(run_state, "metadata", {}).get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    )


def _effective_runtime_hash(run_state: Any) -> str | None:
    """Return the effective-runtime hash from run-state metadata when present."""
    value = getattr(run_state, "metadata", {}).get(
        BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY
    )
    return value if isinstance(value, str) else None


def _tree_branch_limit_input_value(
    *,
    runtime_control: MorpionBootstrapRuntimeControl,
    baseline_tree_branch_limit: int | None,
) -> int:
    """Return the displayed tree-branch-limit value for the dashboard input."""
    if runtime_control.tree_branch_limit is not None:
        return runtime_control.tree_branch_limit
    if baseline_tree_branch_limit is not None:
        return baseline_tree_branch_limit
    return DEFAULT_MORPION_TREE_BRANCH_LIMIT


def _field_status_summary(
    *,
    baseline: object | None,
    current_override: object | None,
    applied_override: object | None,
    effective: object | None,
) -> dict[str, object | None]:
    """Return one stable four-layer control-state summary."""
    return {
        "baseline": baseline,
        "current_override": current_override,
        "applied_override": applied_override,
        "effective": effective,
    }


def _effective_control_value(
    *,
    baseline: object | None,
    override: object | None,
) -> object | None:
    """Return the control value currently in force for one baseline/override pair."""
    return baseline if override is None else override


def _dataset_status_summary(
    config: MorpionBootstrapConfig | None,
    current_control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> dict[str, dict[str, object | None]]:
    """Return one stable dataset-control status summary."""
    baseline_max_rows = None if config is None else config.dataset.max_rows
    baseline_use_backed_up_value = (
        None if config is None else config.dataset.use_backed_up_value
    )
    return {
        "max_rows": _field_status_summary(
            baseline=baseline_max_rows,
            current_override=current_control.max_rows,
            applied_override=applied_control.max_rows,
            effective=_effective_control_value(
                baseline=baseline_max_rows,
                override=applied_control.max_rows,
            ),
        ),
        "use_backed_up_value": _field_status_summary(
            baseline=baseline_use_backed_up_value,
            current_override=current_control.use_backed_up_value,
            applied_override=applied_control.use_backed_up_value,
            effective=_effective_control_value(
                baseline=baseline_use_backed_up_value,
                override=applied_control.use_backed_up_value,
            ),
        ),
    }


def _scheduling_status_summary(
    config: MorpionBootstrapConfig | None,
    current_control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> dict[str, dict[str, object | None]]:
    """Return one stable scheduling-control status summary."""
    baseline_growth_steps = (
        None if config is None else config.runtime.max_growth_steps_per_cycle
    )
    baseline_save_after_seconds = (
        None if config is None else config.runtime.save_after_seconds
    )
    baseline_growth_factor = (
        None if config is None else config.runtime.save_after_tree_growth_factor
    )
    return {
        "max_growth_steps_per_cycle": _field_status_summary(
            baseline=baseline_growth_steps,
            current_override=current_control.max_growth_steps_per_cycle,
            applied_override=applied_control.max_growth_steps_per_cycle,
            effective=_effective_control_value(
                baseline=baseline_growth_steps,
                override=applied_control.max_growth_steps_per_cycle,
            ),
        ),
        "save_after_seconds": _field_status_summary(
            baseline=baseline_save_after_seconds,
            current_override=current_control.save_after_seconds,
            applied_override=applied_control.save_after_seconds,
            effective=_effective_control_value(
                baseline=baseline_save_after_seconds,
                override=applied_control.save_after_seconds,
            ),
        ),
        "save_after_tree_growth_factor": _field_status_summary(
            baseline=baseline_growth_factor,
            current_override=current_control.save_after_tree_growth_factor,
            applied_override=applied_control.save_after_tree_growth_factor,
            effective=_effective_control_value(
                baseline=baseline_growth_factor,
                override=applied_control.save_after_tree_growth_factor,
            ),
        ),
    }


def _is_stale_forced_evaluator(
    forced_evaluator: str | None,
    configured_evaluator_names: tuple[str, ...],
) -> bool:
    """Return whether one forced evaluator is absent from current config."""
    return (
        forced_evaluator is not None
        and forced_evaluator not in configured_evaluator_names
    )


def _evaluator_control_status_summary(
    *,
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object | dict[str, object | None]]:
    """Return one stable evaluator-control status summary."""
    current_mode = "auto" if control.force_evaluator is None else "forced"
    applied_mode = "auto" if applied_control.force_evaluator is None else "forced"
    return {
        "selection_mode": _field_status_summary(
            baseline="auto",
            current_override=current_mode,
            applied_override=applied_mode,
            effective=applied_mode,
        ),
        "forced_evaluator": _field_status_summary(
            baseline=None,
            current_override=control.force_evaluator,
            applied_override=applied_control.force_evaluator,
            effective=applied_control.force_evaluator,
        ),
        "current_force_evaluator_is_stale": _is_stale_forced_evaluator(
            control.force_evaluator,
            configured_evaluator_names,
        ),
        "applied_force_evaluator_is_stale": _is_stale_forced_evaluator(
            applied_control.force_evaluator,
            configured_evaluator_names,
        ),
    }


def _runtime_status_summary(
    *,
    baseline_tree_branch_limit: int,
    current_runtime_control: MorpionBootstrapRuntimeControl,
    applied_runtime_control: MorpionBootstrapRuntimeControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    effective_runtime_hash: str | None,
) -> dict[str, object | dict[str, object | None]]:
    """Return one stable runtime-control status summary."""
    return {
        "tree_branch_limit": _field_status_summary(
            baseline=baseline_tree_branch_limit,
            current_override=current_runtime_control.tree_branch_limit,
            applied_override=applied_runtime_control.tree_branch_limit,
            effective=None
            if effective_runtime_config is None
            else effective_runtime_config.tree_branch_limit,
        ),
        "effective_runtime_hash": effective_runtime_hash,
    }


def _pending_control_fields(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> tuple[str, ...]:
    """Return stable field names that still differ from the last applied control."""
    pending_fields: list[str] = []
    if control.max_rows != applied_control.max_rows:
        pending_fields.append("max_rows")
    if control.use_backed_up_value != applied_control.use_backed_up_value:
        pending_fields.append("use_backed_up_value")
    if control.max_growth_steps_per_cycle != applied_control.max_growth_steps_per_cycle:
        pending_fields.append("max_growth_steps_per_cycle")
    if control.save_after_seconds != applied_control.save_after_seconds:
        pending_fields.append("save_after_seconds")
    if (
        control.save_after_tree_growth_factor
        != applied_control.save_after_tree_growth_factor
    ):
        pending_fields.append("save_after_tree_growth_factor")
    if control.force_evaluator != applied_control.force_evaluator:
        pending_fields.append("force_evaluator")
    if control.runtime.tree_branch_limit != applied_control.runtime.tree_branch_limit:
        pending_fields.append("runtime.tree_branch_limit")
    return tuple(pending_fields)


def _pending_control_sections(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> tuple[str, ...]:
    """Return stable control-section names containing unapplied changes."""
    fields = set(_pending_control_fields(control, applied_control))
    sections: list[str] = []
    if {"max_rows", "use_backed_up_value"} & fields:
        sections.append("dataset")
    if {
        "max_growth_steps_per_cycle",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    } & fields:
        sections.append("scheduling")
    if "force_evaluator" in fields:
        sections.append("evaluator selection")
    if "runtime.tree_branch_limit" in fields:
        sections.append("runtime")
    return tuple(sections)


def _effective_state_summary(
    *,
    run_summary: Any,
    run_state: Any,
    current_control: MorpionBootstrapControl,
    baseline_tree_branch_limit: int,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    latest_dataset_rows: object | None,
    pending_changes: bool,
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object | None]:
    """Return one compact operator-facing effective-state summary."""
    active_evaluator = getattr(run_summary, "latest_active_evaluator_name", None)
    if active_evaluator is None:
        active_evaluator = getattr(run_state, "active_evaluator_name", None)
    evaluator_set_summary = _evaluator_set_summary(configured_evaluator_names)
    return {
        "active_evaluator": active_evaluator,
        "forced_evaluator_request": current_control.force_evaluator,
        "forced_evaluator_request_label": _format_force_evaluator_state(
            current_control.force_evaluator,
            configured_evaluator_names=configured_evaluator_names,
        ),
        "baseline_tree_branch_limit": baseline_tree_branch_limit,
        "effective_tree_branch_limit": None
        if effective_runtime_config is None
        else effective_runtime_config.tree_branch_limit,
        "runtime_override_status": "set"
        if current_control.runtime.tree_branch_limit is not None
        else "unset",
        "evaluator_set_label": evaluator_set_summary["label"],
        "configured_evaluator_count": evaluator_set_summary["count"],
        "configured_evaluator_names": evaluator_set_summary[
            "configured_evaluator_names"
        ],
        "is_canonical_evaluator_family": evaluator_set_summary["is_canonical_family"],
        "latest_dataset_rows": latest_dataset_rows,
        "control_pending_application": pending_changes,
    }


def _summary_layer(
    summary: dict[str, dict[str, object | None]],
    layer: str,
) -> dict[str, object | None]:
    """Return one stable status layer extracted from a field summary."""
    return {
        field_name: field_summary[layer]
        for field_name, field_summary in summary.items()
    }


def _render_pending_changes_section(
    *,
    st: Any,
    pending_changes: bool,
    pending_sections: tuple[str, ...],
    pending_fields: tuple[str, ...],
) -> None:
    """Render one compact summary of unapplied control changes."""
    st.subheader("Pending Changes")
    if pending_changes:
        st.warning("Pending changes will apply at the next cycle boundary.")
        st.write("Pending sections:", pending_sections)
        st.write("Pending fields:", pending_fields)
    else:
        st.success("Control file matches the last applied cycle boundary state.")
        st.write("Pending sections:", ())
        st.write("Pending fields:", ())


def _render_effective_state_section(
    *,
    st: Any,
    summary: dict[str, object | None],
) -> None:
    """Render one compact operator-facing effective-state panel."""
    st.subheader("Effective State")
    columns = st.columns(4)
    columns[0].metric("Active evaluator", _format_value(summary["active_evaluator"]))
    columns[1].metric(
        "Forced evaluator request",
        _format_value(summary["forced_evaluator_request_label"]),
    )
    columns[2].metric(
        "Effective tree branch limit",
        _format_value(summary["effective_tree_branch_limit"]),
    )
    columns[3].metric(
        "Pending control file",
        "yes" if bool(summary["control_pending_application"]) else "no",
    )
    st.write("Evaluator set:", _format_value(summary["evaluator_set_label"]))
    st.write(
        "Configured evaluators:",
        _format_value(summary["configured_evaluator_names"]),
    )
    st.write("Summary:", summary)


def _evaluator_set_summary(
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object]:
    """Return one compact evaluator-set summary for dashboard/operator views."""
    sorted_names = tuple(sorted(configured_evaluator_names))
    canonical_names = tuple(sorted(canonical_morpion_evaluator_names()))
    is_canonical_family = sorted_names == canonical_names
    if not sorted_names:
        label = "no configured evaluators"
    elif is_canonical_family:
        label = "canonical 8-model family"
    else:
        label = f"custom ({len(sorted_names)} evaluators)"
    return {
        "label": label,
        "count": len(sorted_names),
        "configured_evaluator_names": sorted_names,
        "is_canonical_family": is_canonical_family,
    }


def _render_status_layers(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render baseline/current/applied/effective layers for one control section."""
    columns = st.columns(4)
    columns[0].write("Baseline")
    columns[0].write(_summary_layer(summary, "baseline"))
    columns[1].write("Current override")
    columns[1].write(_summary_layer(summary, "current_override"))
    columns[2].write("Last applied")
    columns[2].write(_summary_layer(summary, "applied_override"))
    columns[3].write("Effective")
    columns[3].write(_summary_layer(summary, "effective"))


def _render_dataset_control_section(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render the dataset-control summary block."""
    st.subheader("Dataset / Training Data Extraction")
    st.caption("Unchecked inputs inherit the dataset baseline from persisted config.")
    _render_status_layers(st=st, summary=summary)


def _render_scheduling_control_section(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render the cycle-scheduling control summary block."""
    st.subheader("Cycle Scheduling")
    st.caption(
        "Scheduling overrides are persisted in the control file and apply at cycle boundaries."
    )
    _render_status_layers(st=st, summary=summary)


def _render_evaluator_control_section(
    *,
    st: Any,
    summary: dict[str, object | dict[str, object | None]],
) -> None:
    """Render the evaluator-selection control summary block."""
    st.subheader("Evaluator Selection")
    st.caption(
        "Auto keeps normal evaluator selection. Forced persists one explicit evaluator name."
    )
    _render_status_layers(
        st=st,
        summary={
            "selection_mode": summary["selection_mode"],
            "forced_evaluator": summary["forced_evaluator"],
        },
    )
    st.write(
        "Stale forced evaluator flags:",
        {
            "current": summary["current_force_evaluator_is_stale"],
            "applied": summary["applied_force_evaluator_is_stale"],
        },
    )


def _render_runtime_control_section(
    *,
    st: Any,
    summary: dict[str, object | dict[str, object | None]],
) -> None:
    """Render the runtime-control summary block."""
    st.subheader("Runtime Control")
    st.caption(
        "Baseline comes from persisted config, override comes from the control file, "
        "and effective runtime reflects what the loop has actually applied. On an "
        "existing persisted tree, only non-increasing tree branch limit changes are supported."
    )
    _render_status_layers(
        st=st,
        summary={"tree_branch_limit": summary["tree_branch_limit"]},
    )
    st.write(
        "Effective runtime hash:", _format_value(summary["effective_runtime_hash"])
    )


def _configured_evaluator_names(
    config: MorpionBootstrapConfig | None,
) -> tuple[str, ...]:
    """Return configured evaluator names from persisted bootstrap config."""
    if config is None:
        return ()
    return tuple(config.evaluators.evaluators)


def _force_evaluator_options(
    *,
    configured_evaluator_names: tuple[str, ...],
    current_force_evaluator: str | None,
) -> tuple[str, ...]:
    """Return selectable forced evaluators from config, preserving any stale current value."""
    options = list(configured_evaluator_names)
    if current_force_evaluator is not None and current_force_evaluator not in options:
        options.append(current_force_evaluator)
    return tuple(options)


def _has_pending_control_changes(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> bool:
    """Return whether the control file differs from the last applied control."""
    return control != applied_control


def _build_next_control(
    *,
    override_max_growth_steps_per_cycle: bool,
    max_growth_steps_per_cycle: int,
    override_max_rows: bool,
    max_rows: int,
    override_use_backed_up_value: bool,
    use_backed_up_value: bool,
    override_save_after_seconds: bool,
    save_after_seconds: float,
    override_save_after_tree_growth_factor: bool,
    save_after_tree_growth_factor: float,
    override_tree_branch_limit: bool,
    tree_branch_limit: int,
    force_evaluator_mode: str,
    force_evaluator: str,
) -> MorpionBootstrapControl:
    """Build one persisted control payload from tri-state dashboard inputs."""
    return MorpionBootstrapControl(
        max_growth_steps_per_cycle=max_growth_steps_per_cycle
        if override_max_growth_steps_per_cycle
        else None,
        max_rows=max_rows if override_max_rows else None,
        use_backed_up_value=use_backed_up_value
        if override_use_backed_up_value
        else None,
        save_after_seconds=save_after_seconds if override_save_after_seconds else None,
        save_after_tree_growth_factor=save_after_tree_growth_factor
        if override_save_after_tree_growth_factor
        else None,
        force_evaluator=_resolved_force_evaluator(
            force_evaluator_mode=force_evaluator_mode,
            force_evaluator=force_evaluator,
        ),
        runtime=MorpionBootstrapRuntimeControl(
            tree_branch_limit=tree_branch_limit if override_tree_branch_limit else None
        ),
    )


def _latest_optional_value(series: tuple[Any, ...]) -> object | None:
    """Return the latest value from one optional dashboard series."""
    if not series:
        return None
    return getattr(series[-1], "value", None)


def _format_value(value: object | None) -> str:
    """Render optional values consistently in the dashboard."""
    return "n/a" if value is None else str(value)


def _format_bool_icon(value: bool | None) -> str:
    """Render one optional boolean with compact visual icons."""
    if value is True:
        return "✔"
    if value is False:
        return "✖"
    return "—"


def _format_optional_runtime_override(value: int | None) -> str:
    """Render one optional runtime override for human dashboard display."""
    return "unset" if value is None else str(value)


def _force_evaluator_option_index(
    options: tuple[str, ...],
    current_force_evaluator: str | None,
) -> int:
    """Return the current forced-evaluator index for one select widget."""
    if not options:
        return 0
    if current_force_evaluator is None:
        return 0
    try:
        return options.index(current_force_evaluator)
    except ValueError:
        return 0


def _format_force_evaluator_state(
    value: str | None,
    *,
    configured_evaluator_names: tuple[str, ...],
) -> str:
    """Render one requested forced evaluator state for operator display."""
    if value is None:
        return "auto"
    if _is_stale_forced_evaluator(value, configured_evaluator_names):
        return f"{value} (stale / not configured)"
    return value


def _format_force_evaluator_option(
    value: str,
    *,
    configured_evaluator_names: tuple[str, ...] = (),
) -> str:
    """Render one forced-evaluator option for the Streamlit select widget."""
    if not value:
        return "No configured evaluators"
    if _is_stale_forced_evaluator(
        value,
        configured_evaluator_names,
    ):
        return f"{value} (stale / not configured)"
    return value


def _control_number_value(value: int | None, *, default: int = 0) -> int:
    """Return one Streamlit-safe integer input default."""
    return default if value is None else value


def _control_float_value(value: float | None, *, default: float = 0.0) -> float:
    """Return one Streamlit-safe float input default."""
    return default if value is None else value


def _resolved_force_evaluator(
    *,
    force_evaluator_mode: str,
    force_evaluator: str,
) -> str | None:
    """Resolve the tri-state dashboard force-evaluator controls into persisted data."""
    if force_evaluator_mode != "forced" or not force_evaluator:
        return None
    return force_evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    cli_args = parser.parse_args()
    run_dashboard_app(cli_args.work_dir)


__all__ = ["run_dashboard_app"]
