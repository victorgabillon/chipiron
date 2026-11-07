"""Exceptional shutdown uses bounded owned cleanup even when discovery fails."""

import json
import signal
import time
from typing import TYPE_CHECKING, Any

import psutil
import pytest

from tests.environments.test_morpion_persistent_shutdown import _owned_deadline

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.fast
def test_cleanup_discovery_exception_reaps_owned_processes_and_restores_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed discovery cannot strand Popen.__exit__ waiting on owned writers."""
    before = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    original_children = psutil.Process.children
    real_monotonic, real_sleep = time.monotonic, time.sleep

    def children(process: Any, recursive: bool = False) -> Any:
        if not recursive:
            message = "injected cleanup discovery failure"
            raise RuntimeError(message)
        return original_children(process, recursive=True)

    monkeypatch.setattr(psutil.Process, "children", children)
    with pytest.raises(RuntimeError, match="injected cleanup discovery failure"):
        _owned_deadline(tmp_path, monkeypatch, 143)
    assert all(
        signal.getsignal(number) == handler for number, handler in before.items()
    )
    decision_path = next((tmp_path / "resource_monitor").glob("decision_*.json"))
    assert json.loads(decision_path.read_text())["reason"] == "supervisor_deadline"
    pid = int((tmp_path / "ready").read_text())
    deadline = real_monotonic() + 3
    while psutil.pid_exists(pid) and real_monotonic() < deadline:
        real_sleep(0.01)
    assert not psutil.pid_exists(pid)
