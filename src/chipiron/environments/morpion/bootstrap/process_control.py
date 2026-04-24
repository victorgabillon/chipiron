"""Thin file-backed launcher process control for the Morpion dashboard."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bootstrap_loop import MorpionBootstrapPaths


@dataclass(frozen=True, slots=True)
class MorpionBootstrapProcessState:
    """Persisted launcher-process state reconciled with live PID status."""

    pid: int | None
    is_running: bool
    command: tuple[str, ...]
    work_dir: str
    started_at_utc: str | None = None
    stopped_at_utc: str | None = None
    last_exit_code: int | None = None
    last_stop_reason: str | None = None
    status_label: str = "stopped"


@dataclass(frozen=True, slots=True)
class MorpionBootstrapStartResult:
    """Result of one launcher-process start attempt."""

    state: MorpionBootstrapProcessState
    already_running: bool


@dataclass(frozen=True, slots=True)
class MorpionBootstrapStopResult:
    """Result of one launcher-process stop attempt."""

    state: MorpionBootstrapProcessState
    was_running: bool


class MorpionBootstrapProcessControlError(RuntimeError):
    """Base error for Morpion launcher process-control failures."""

    @classmethod
    def exited_immediately(
        cls, *, stdout_path: Path, stderr_path: Path
    ) -> MorpionBootstrapProcessControlError:
        """Return the launcher-start failure when the process exits immediately."""
        return cls(
            "Launcher process exited immediately after start. "
            f"Check logs:\nstdout={stdout_path}\n"
            f"stderr={stderr_path}"
        )

    @classmethod
    def terminate_failed(
        cls, *, pid: int, error: OSError
    ) -> MorpionBootstrapProcessControlError:
        """Return the launcher terminate failure."""
        return cls(f"Failed to terminate launcher process {pid}: {error}")

    @classmethod
    def kill_failed(
        cls, *, pid: int, error: OSError
    ) -> MorpionBootstrapProcessControlError:
        """Return the launcher kill failure."""
        return cls(f"Failed to kill launcher process {pid}: {error}")


class MorpionBootstrapProcessAlreadyRunningError(MorpionBootstrapProcessControlError):
    """Raised when a start request targets an already-running launcher."""


class MorpionBootstrapProcessNotRunningError(MorpionBootstrapProcessControlError):
    """Raised when a stop request targets no running launcher."""


def launcher_command_for_work_dir(work_dir: str | Path) -> tuple[str, ...]:
    """Return the canonical launcher command for one work dir."""
    resolved_work_dir = Path(work_dir).resolve()
    return (
        sys.executable,
        "-m",
        "chipiron.environments.morpion.bootstrap.launcher",
        "--work-dir",
        str(resolved_work_dir),
    )


def load_morpion_bootstrap_process_state(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapProcessState:
    """Load the persisted launcher-process state and reconcile it with the live PID."""
    raw_state = _read_process_state_file(paths.launcher_process_state_path) or {}
    pid = _read_pid_file(paths.launcher_pid_path)
    command = _coerce_command(
        raw_state.get("command"),
        default=launcher_command_for_work_dir(paths.work_dir),
    )
    started_at_utc = _coerce_optional_str(raw_state.get("started_at_utc"))
    stopped_at_utc = _coerce_optional_str(raw_state.get("stopped_at_utc"))
    last_exit_code = _coerce_optional_int(raw_state.get("last_exit_code"))
    last_stop_reason = _coerce_optional_str(raw_state.get("last_stop_reason"))
    if pid is not None and _pid_is_alive(pid):
        return MorpionBootstrapProcessState(
            pid=pid,
            is_running=True,
            command=command,
            work_dir=str(paths.work_dir),
            started_at_utc=started_at_utc,
            stopped_at_utc=None,
            last_exit_code=last_exit_code,
            last_stop_reason=last_stop_reason,
            status_label="running",
        )
    if pid is not None:
        return MorpionBootstrapProcessState(
            pid=None,
            is_running=False,
            command=command,
            work_dir=str(paths.work_dir),
            started_at_utc=started_at_utc,
            stopped_at_utc=stopped_at_utc,
            last_exit_code=last_exit_code,
            last_stop_reason=last_stop_reason,
            status_label="stale pid",
        )
    return _stopped_process_state(
        paths,
        command=command,
        started_at_utc=started_at_utc,
        stopped_at_utc=stopped_at_utc,
        last_exit_code=last_exit_code,
        last_stop_reason=last_stop_reason,
    )


def start_morpion_bootstrap_process(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapStartResult:
    """Start the canonical launcher subprocess for this work dir if not already running."""
    current_state = load_morpion_bootstrap_process_state(paths)
    if current_state.is_running:
        return MorpionBootstrapStartResult(state=current_state, already_running=True)

    command = launcher_command_for_work_dir(paths.work_dir)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with (
        paths.launcher_stdout_log_path.open("ab") as stdout_file,
        paths.launcher_stderr_log_path.open("ab") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

    time.sleep(0.2)
    if process.poll() is not None:
        raise MorpionBootstrapProcessControlError.exited_immediately(
            stdout_path=paths.launcher_stdout_log_path,
            stderr_path=paths.launcher_stderr_log_path,
        )

    _write_pid_file(paths.launcher_pid_path, process.pid)
    state = MorpionBootstrapProcessState(
        pid=process.pid,
        is_running=True,
        command=command,
        work_dir=str(paths.work_dir),
        started_at_utc=_utc_now_iso(),
        stopped_at_utc=None,
        last_exit_code=None,
        last_stop_reason=None,
        status_label="running",
    )
    _write_process_state_file(paths.launcher_process_state_path, state)
    return MorpionBootstrapStartResult(state=state, already_running=False)


def stop_morpion_bootstrap_process(
    paths: MorpionBootstrapPaths,
    *,
    reason: str = "dashboard_stop",
    timeout_s: float = 5.0,
) -> MorpionBootstrapStopResult:
    """Stop the launcher subprocess for this work dir if it is running."""
    current_state = load_morpion_bootstrap_process_state(paths)
    if not current_state.is_running or current_state.pid is None:
        stopped_state = _stopped_process_state(
            paths,
            command=current_state.command,
            started_at_utc=current_state.started_at_utc,
            stopped_at_utc=current_state.stopped_at_utc,
            last_exit_code=current_state.last_exit_code,
            last_stop_reason=current_state.last_stop_reason,
            status_label=current_state.status_label,
        )
        _remove_pid_file(paths.launcher_pid_path)
        _write_process_state_file(paths.launcher_process_state_path, stopped_state)
        return MorpionBootstrapStopResult(state=stopped_state, was_running=False)

    pid = current_state.pid
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise MorpionBootstrapProcessControlError.terminate_failed(
            pid=pid, error=exc
        ) from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            break
        time.sleep(0.05)
    if _pid_is_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise MorpionBootstrapProcessControlError.kill_failed(
                pid=pid, error=exc
            ) from exc
        second_deadline = time.monotonic() + 1.0
        while time.monotonic() < second_deadline and _pid_is_alive(pid):
            time.sleep(0.05)

    _remove_pid_file(paths.launcher_pid_path)
    stopped_state = _stopped_process_state(
        paths,
        command=current_state.command,
        started_at_utc=current_state.started_at_utc,
        stopped_at_utc=_utc_now_iso(),
        last_exit_code=None,
        last_stop_reason=reason,
    )
    _write_process_state_file(paths.launcher_process_state_path, stopped_state)
    return MorpionBootstrapStopResult(state=stopped_state, was_running=True)


def restart_morpion_bootstrap_process(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapProcessState:
    """Restart the launcher subprocess for this work dir."""
    stop_morpion_bootstrap_process(paths, reason="dashboard_restart")
    return start_morpion_bootstrap_process(paths).state


def register_current_launcher_process(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapProcessState:
    """Register the current process as the active launcher for this work dir."""
    state = MorpionBootstrapProcessState(
        pid=os.getpid(),
        is_running=True,
        command=launcher_command_for_work_dir(paths.work_dir),
        work_dir=str(paths.work_dir),
        started_at_utc=_utc_now_iso(),
        stopped_at_utc=None,
        last_exit_code=None,
        last_stop_reason=None,
        status_label="running",
    )
    _write_pid_file(paths.launcher_pid_path, state.pid)
    _write_process_state_file(paths.launcher_process_state_path, state)
    return state


def mark_current_launcher_process_stopped(
    paths: MorpionBootstrapPaths,
    *,
    exit_code: int | None = None,
    reason: str = "launcher_exit",
) -> MorpionBootstrapProcessState:
    """Mark the current launcher process as stopped and clear PID ownership."""
    current_state = load_morpion_bootstrap_process_state(paths)
    stopped_state = _stopped_process_state(
        paths,
        command=current_state.command,
        started_at_utc=current_state.started_at_utc,
        stopped_at_utc=_utc_now_iso(),
        last_exit_code=exit_code,
        last_stop_reason=reason,
    )
    _remove_pid_file(paths.launcher_pid_path)
    _write_process_state_file(paths.launcher_process_state_path, stopped_state)
    return stopped_state


def _pid_is_alive(pid: int) -> bool:
    """Return whether one POSIX process id currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid_file(path: Path) -> int | None:
    """Read one launcher pid file tolerantly."""
    if not path.is_file():
        return None
    try:
        raw_value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_value:
        return None
    try:
        pid = int(raw_value)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _write_pid_file(path: Path, pid: int) -> None:
    """Write one launcher pid file."""
    path.write_text(f"{pid}\n", encoding="utf-8")


def _remove_pid_file(path: Path) -> None:
    """Remove one launcher pid file if present."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _read_process_state_file(path: Path) -> dict[str, object] | None:
    """Read one process-state json file tolerantly."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_process_state_file(path: Path, state: MorpionBootstrapProcessState) -> None:
    """Persist one launcher-process state as json."""
    payload = {
        "pid": state.pid,
        "is_running": state.is_running,
        "command": list(state.command),
        "work_dir": state.work_dir,
        "started_at_utc": state.started_at_utc,
        "stopped_at_utc": state.stopped_at_utc,
        "last_exit_code": state.last_exit_code,
        "last_stop_reason": state.last_stop_reason,
        "status_label": state.status_label,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO 8601 form."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stopped_process_state(
    paths: MorpionBootstrapPaths,
    *,
    command: tuple[str, ...] | None = None,
    started_at_utc: str | None = None,
    stopped_at_utc: str | None = None,
    last_exit_code: int | None = None,
    last_stop_reason: str | None = None,
    status_label: str = "stopped",
) -> MorpionBootstrapProcessState:
    """Build one stopped launcher-process state."""
    return MorpionBootstrapProcessState(
        pid=None,
        is_running=False,
        command=launcher_command_for_work_dir(paths.work_dir)
        if command is None
        else command,
        work_dir=str(paths.work_dir),
        started_at_utc=started_at_utc,
        stopped_at_utc=stopped_at_utc,
        last_exit_code=last_exit_code,
        last_stop_reason=last_stop_reason,
        status_label=status_label,
    )


def _coerce_command(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Return one persisted command tuple or a stable default."""
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return tuple(value)
    if isinstance(value, tuple) and all(isinstance(part, str) for part in value):
        return value
    return default


def _coerce_optional_str(value: object) -> str | None:
    """Return one optional string from a tolerant persisted payload."""
    return value if isinstance(value, str) else None


def _coerce_optional_int(value: object) -> int | None:
    """Return one optional integer from a tolerant persisted payload."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "MorpionBootstrapProcessAlreadyRunningError",
    "MorpionBootstrapProcessControlError",
    "MorpionBootstrapProcessNotRunningError",
    "MorpionBootstrapProcessState",
    "MorpionBootstrapStartResult",
    "MorpionBootstrapStopResult",
    "launcher_command_for_work_dir",
    "load_morpion_bootstrap_process_state",
    "mark_current_launcher_process_stopped",
    "register_current_launcher_process",
    "restart_morpion_bootstrap_process",
    "start_morpion_bootstrap_process",
    "stop_morpion_bootstrap_process",
]
