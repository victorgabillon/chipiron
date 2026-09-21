"""Dashboard launcher process controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.process_control import (
    MorpionBootstrapProcessControlError,
    MorpionBootstrapProcessState,
    launcher_command_for_work_dir,
    load_morpion_bootstrap_process_state,
    restart_morpion_bootstrap_process,
    start_morpion_bootstrap_process,
    stop_morpion_bootstrap_process,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

__all__ = [
    "render_launcher_command_text",
    "render_run_control_section",
    "render_run_control_state",
]


def render_run_control_section(*, st: Any, paths: MorpionBootstrapPaths) -> None:
    """Render Start / Stop / Restart controls for the launcher subprocess."""
    st.subheader("Run Control")
    state = load_morpion_bootstrap_process_state(paths)
    render_run_control_state(st=st, paths=paths, state=state)

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


def render_run_control_state(
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
        render_launcher_command_text(
            state.command or launcher_command_for_work_dir(paths.work_dir)
        )
    )
    st.write("stdout log:", str(paths.launcher_stdout_log_path))
    st.write("stderr log:", str(paths.launcher_stderr_log_path))


def render_launcher_command_text(command: tuple[str, ...]) -> str:
    """Render one launcher command tuple as a stable shell-style string."""
    return " ".join(command)
