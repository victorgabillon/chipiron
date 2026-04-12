"""Local Streamlit dashboard for monitoring and controlling Morpion bootstrap runs."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from .bootstrap_loop import MorpionBootstrapPaths
from .control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    MorpionBootstrapControl,
    bootstrap_control_from_metadata,
    bootstrap_control_to_dict,
    load_bootstrap_control,
    save_bootstrap_control,
)
from .dashboard_plot import (
    plot_active_evaluator,
    plot_dataset_size,
    plot_evaluator_losses,
    plot_tree_size,
)
from .history_view import build_morpion_bootstrap_dashboard_data
from .run_state import initialize_bootstrap_run_state, load_bootstrap_run_state


def run_dashboard_app(work_dir: Path) -> None:
    """Render the local Streamlit dashboard for one bootstrap work directory."""
    st = _get_streamlit()
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    control = load_bootstrap_control(paths.control_path)
    applied_control = _load_applied_control(paths)
    dashboard_data = build_morpion_bootstrap_dashboard_data(paths.work_dir)
    run_state = _load_run_state(paths)
    pending_changes = control != applied_control

    st.set_page_config(page_title="Morpion Bootstrap Dashboard", layout="wide")
    st.title("Morpion Bootstrap Dashboard")
    st.caption(str(paths.work_dir))

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

    st.subheader("Controls")
    if pending_changes:
        st.warning("Pending changes will apply at the next cycle boundary.")
    else:
        st.success("Control file matches the last applied cycle boundary state.")

    control_options = ["auto", *sorted(dashboard_data.evaluator_loss_by_name)]
    current_force_evaluator = (
        "auto" if control.force_evaluator is None else control.force_evaluator
    )
    if current_force_evaluator not in control_options:
        control_options.append(current_force_evaluator)

    with st.form("bootstrap-controls"):
        max_growth_steps_per_cycle = st.number_input(
            "Max growth steps per cycle",
            min_value=0,
            value=_control_number_value(control.max_growth_steps_per_cycle),
        )
        max_rows = st.number_input(
            "Max rows",
            min_value=0,
            value=_control_number_value(control.max_rows),
        )
        use_backed_up_value = st.checkbox(
            "Use backed-up value",
            value=False if control.use_backed_up_value is None else control.use_backed_up_value,
        )
        save_after_seconds = st.number_input(
            "Save after seconds",
            min_value=0.0,
            value=_control_float_value(control.save_after_seconds),
            step=1.0,
        )
        save_after_tree_growth_factor = st.number_input(
            "Save after tree growth factor",
            min_value=0.0,
            value=_control_float_value(control.save_after_tree_growth_factor, default=2.0),
            step=0.1,
        )
        force_evaluator = st.selectbox(
            "Force evaluator",
            options=control_options,
            index=control_options.index(current_force_evaluator),
        )
        if st.form_submit_button("Apply changes"):
            next_control = MorpionBootstrapControl(
                max_growth_steps_per_cycle=max_growth_steps_per_cycle,
                max_rows=max_rows,
                use_backed_up_value=use_backed_up_value,
                save_after_seconds=save_after_seconds,
                save_after_tree_growth_factor=save_after_tree_growth_factor,
                force_evaluator=None if force_evaluator == "auto" else force_evaluator,
            )
            save_bootstrap_control(next_control, paths.control_path)
            st.success("Saved control changes. They will apply at the next cycle boundary.")

    st.subheader("Plots")
    plot_columns = st.columns(2)
    with plot_columns[0]:
        _render_plot(st, lambda: plot_tree_size(dashboard_data.tree_num_nodes))
        _render_plot(st, lambda: plot_dataset_size(dashboard_data.dataset_num_rows))
    with plot_columns[1]:
        _render_plot(st, lambda: plot_active_evaluator(dashboard_data.active_evaluator))
        _render_plot(
            st,
            lambda: plot_evaluator_losses(dashboard_data.evaluator_loss_by_name),
        )

    st.subheader("Debug Info")
    st.write("Config hash:", _format_value(run_state.metadata.get("bootstrap_config_hash")))
    st.write(
        "Last checkpoint path:",
        _format_value(run_state.metadata.get("runtime_checkpoint_path")),
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
        raise RuntimeError(
            "Streamlit is not installed. Install `streamlit` to use the local dashboard."
        ) from exc


def _render_plot(st: Any, build_plot: Any) -> None:
    """Render one existing matplotlib plot helper into Streamlit."""
    build_plot()
    figure = plt.gcf()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


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


def _latest_optional_value(series: tuple[Any, ...]) -> object | None:
    """Return the latest value from one optional dashboard series."""
    if not series:
        return None
    return getattr(series[-1], "value", None)


def _format_value(value: object | None) -> str:
    """Render optional values consistently in the dashboard."""
    return "n/a" if value is None else str(value)


def _control_number_value(value: int | None) -> int:
    """Return one Streamlit-safe integer input default."""
    return 0 if value is None else value


def _control_float_value(value: float | None, *, default: float = 0.0) -> float:
    """Return one Streamlit-safe float input default."""
    return default if value is None else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    cli_args = parser.parse_args()
    run_dashboard_app(cli_args.work_dir)


__all__ = ["run_dashboard_app"]