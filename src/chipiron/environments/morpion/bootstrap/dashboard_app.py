"""Local Streamlit dashboard for monitoring and controlling Morpion bootstrap runs."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from .bootstrap_loop import MorpionBootstrapPaths
from .config import load_bootstrap_config
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
    plot_record_score,
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
    pending_changes = _has_pending_control_changes(control, applied_control)
    configured_evaluator_names = _configured_evaluator_names(paths)
    force_evaluator_options = _force_evaluator_options(
        configured_evaluator_names=configured_evaluator_names,
        current_force_evaluator=control.force_evaluator,
    )

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
    st.caption("Unchecked overrides leave the persisted bootstrap config unchanged.")

    with st.form("bootstrap-controls"):
        override_max_growth_steps_per_cycle = st.checkbox(
            "Override max growth steps per cycle",
            value=control.max_growth_steps_per_cycle is not None,
        )
        max_growth_steps_per_cycle = st.number_input(
            "Max growth steps per cycle",
            min_value=0,
            value=_control_number_value(control.max_growth_steps_per_cycle),
            disabled=not override_max_growth_steps_per_cycle,
        )
        override_max_rows = st.checkbox(
            "Override max rows",
            value=control.max_rows is not None,
        )
        max_rows = st.number_input(
            "Max rows",
            min_value=0,
            value=_control_number_value(control.max_rows),
            disabled=not override_max_rows,
        )
        override_use_backed_up_value = st.checkbox(
            "Override use backed-up value",
            value=control.use_backed_up_value is not None,
        )
        use_backed_up_value = st.checkbox(
            "Use backed-up value",
            value=False if control.use_backed_up_value is None else control.use_backed_up_value,
            disabled=not override_use_backed_up_value,
        )
        override_save_after_seconds = st.checkbox(
            "Override save after seconds",
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
            "Override save after tree growth factor",
            value=control.save_after_tree_growth_factor is not None,
        )
        save_after_tree_growth_factor = st.number_input(
            "Save after tree growth factor",
            min_value=0.0,
            value=_control_float_value(control.save_after_tree_growth_factor, default=2.0),
            step=0.1,
            disabled=not override_save_after_tree_growth_factor,
        )
        force_evaluator_mode = st.radio(
            "Evaluator selection mode",
            options=("auto", "forced"),
            index=0 if control.force_evaluator is None else 1,
            horizontal=True,
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
            disabled=(
                force_evaluator_mode != "forced"
                or not force_evaluator_options
            ),
            format_func=_format_force_evaluator_option,
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
                force_evaluator_mode=force_evaluator_mode,
                force_evaluator=force_evaluator,
            )
            save_bootstrap_control(next_control, paths.control_path)
            st.success("Saved control changes. They will apply at the next cycle boundary.")

    st.subheader("Plots")
    plot_columns = st.columns(2)
    with plot_columns[0]:
        _render_plot(st, lambda: plot_tree_size(dashboard_data.tree_num_nodes))
        _render_plot(
            st,
            lambda: plot_record_score(dashboard_data.canonical_record_score),
        )
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


def _configured_evaluator_names(paths: MorpionBootstrapPaths) -> tuple[str, ...]:
    """Return configured evaluator names from persisted bootstrap config."""
    if not paths.bootstrap_config_path.is_file():
        return ()
    config = load_bootstrap_config(paths.bootstrap_config_path)
    return tuple(config.evaluators.evaluators)


def _force_evaluator_options(
    *,
    configured_evaluator_names: tuple[str, ...],
    current_force_evaluator: str | None,
) -> tuple[str, ...]:
    """Return selectable forced evaluators from config, preserving any stale current value."""
    options = list(configured_evaluator_names)
    if (
        current_force_evaluator is not None
        and current_force_evaluator not in options
    ):
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
    )


def _latest_optional_value(series: tuple[Any, ...]) -> object | None:
    """Return the latest value from one optional dashboard series."""
    if not series:
        return None
    return getattr(series[-1], "value", None)


def _format_value(value: object | None) -> str:
    """Render optional values consistently in the dashboard."""
    return "n/a" if value is None else str(value)


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


def _format_force_evaluator_option(value: str) -> str:
    """Render one forced-evaluator option for the Streamlit select widget."""
    return "No configured evaluators" if not value else value


def _control_number_value(value: int | None) -> int:
    """Return one Streamlit-safe integer input default."""
    return 0 if value is None else value


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