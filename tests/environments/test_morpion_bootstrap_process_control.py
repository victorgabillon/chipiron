"""Tests for Morpion launcher process-control helpers."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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

import chipiron.environments.morpion.bootstrap.process_control as process_control_module
from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths


def test_launcher_command_for_work_dir_uses_current_python(tmp_path: Path) -> None:
    """Process-control launches should use the current interpreter and canonical module path."""
    command = process_control_module.launcher_command_for_work_dir(tmp_path)

    assert command[0] == sys.executable
    assert command[1:4] == (
        "-m",
        "chipiron.environments.morpion.bootstrap.launcher",
        "--work-dir",
    )
    assert command[4] == str(tmp_path.resolve())


def test_load_process_state_missing_files_returns_stopped(tmp_path: Path) -> None:
    """Missing pid and state files should produce a tolerant stopped state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    state = process_control_module.load_morpion_bootstrap_process_state(paths)

    assert not state.is_running
    assert state.pid is None
    assert state.status_label == "stopped"


def test_load_process_state_stale_pid_returns_stopped(tmp_path: Path) -> None:
    """A stale pid file should be surfaced as non-running without crashing."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.launcher_pid_path.write_text("999999\n", encoding="utf-8")

    state = process_control_module.load_morpion_bootstrap_process_state(paths)

    assert not state.is_running
    assert state.pid is None
    assert state.status_label == "stale pid"


def test_load_process_state_malformed_json_is_tolerant(tmp_path: Path) -> None:
    """Malformed process-state json should degrade to a stopped state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.launcher_process_state_path.write_text("{not-json", encoding="utf-8")

    state = process_control_module.load_morpion_bootstrap_process_state(paths)

    assert not state.is_running
    assert state.status_label == "stopped"


def test_start_process_writes_pid_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting the launcher should write pid/state files and mark the process running."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    popen_kwargs: dict[str, object] = {}

    def _fake_popen(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        popen_kwargs.update(kwargs)
        return SimpleNamespace(pid=4321, poll=lambda: None)

    monkeypatch.setattr(process_control_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(process_control_module.time, "sleep", lambda _: None)

    result = process_control_module.start_morpion_bootstrap_process(paths)

    assert not result.already_running
    assert result.state.is_running
    assert result.state.pid == 4321
    assert "cwd" not in popen_kwargs
    assert paths.launcher_pid_path.read_text(encoding="utf-8").strip() == "4321"
    assert paths.launcher_process_state_path.is_file()


def test_start_process_raises_when_launcher_exits_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start should fail fast when the launcher subprocess dies before becoming live."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    def _fake_popen(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(pid=4321, poll=lambda: 1)

    monkeypatch.setattr(process_control_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(process_control_module.time, "sleep", lambda _: None)

    with pytest.raises(process_control_module.MorpionBootstrapProcessControlError):
        process_control_module.start_morpion_bootstrap_process(paths)

    assert not paths.launcher_pid_path.exists()
    assert not paths.launcher_process_state_path.exists()


def test_start_when_already_running_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated start should return already_running when the recorded pid is alive."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.launcher_pid_path.write_text("1234\n", encoding="utf-8")
    monkeypatch.setattr(
        process_control_module, "_pid_is_alive", lambda pid: pid == 1234
    )

    result = process_control_module.start_morpion_bootstrap_process(paths)

    assert result.already_running
    assert result.state.is_running
    assert result.state.pid == 1234


def test_stop_when_not_running_is_tolerant(tmp_path: Path) -> None:
    """Stopping with no running launcher should return a stable stopped result."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    result = process_control_module.stop_morpion_bootstrap_process(paths)

    assert not result.was_running
    assert not result.state.is_running


def test_stop_sends_term_and_clears_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop should send SIGTERM, tolerate process exit, and clear pid ownership."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.launcher_pid_path.write_text("1234\n", encoding="utf-8")
    calls: list[tuple[int, int]] = []
    alive_states = iter([True, False, False])

    monkeypatch.setattr(
        process_control_module,
        "_pid_is_alive",
        lambda pid: next(alive_states) if pid == 1234 else False,
    )

    def _fake_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(process_control_module.os, "kill", _fake_kill)

    result = process_control_module.stop_morpion_bootstrap_process(paths)

    assert result.was_running
    assert not result.state.is_running
    assert calls[0] == (1234, signal.SIGTERM)
    assert not paths.launcher_pid_path.exists()


def test_restart_calls_stop_then_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart should delegate to stop then start in order."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    call_order: list[str] = []

    def _fake_stop(*args: object, **kwargs: object) -> object:
        del args, kwargs
        call_order.append("stop")
        return SimpleNamespace(
            state=SimpleNamespace(is_running=False), was_running=True
        )

    def _fake_start(*args: object, **kwargs: object) -> object:
        del args, kwargs
        call_order.append("start")
        return SimpleNamespace(
            state=process_control_module.MorpionBootstrapProcessState(
                pid=55,
                is_running=True,
                command=("python",),
                work_dir=str(paths.work_dir),
                status_label="running",
            ),
            already_running=False,
        )

    monkeypatch.setattr(
        process_control_module, "stop_morpion_bootstrap_process", _fake_stop
    )
    monkeypatch.setattr(
        process_control_module, "start_morpion_bootstrap_process", _fake_start
    )

    state = process_control_module.restart_morpion_bootstrap_process(paths)

    assert call_order == ["stop", "start"]
    assert state.is_running
    assert state.pid == 55
