"""Exceptional shutdown uses bounded owned cleanup even when discovery fails."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from chipiron.environments.morpion.bootstrap import continual_operator as operator
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from tests.environments.test_morpion_persistent_shutdown import _owned_deadline


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


@pytest.mark.fast
@pytest.mark.parametrize(
    "foreign_kind", ["untagged", "session", "ticks", "work_dir", "forged_status"]
)
def test_supervisor_death_before_first_sample_reaps_registered_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_kind: str
) -> None:
    """A real killed Bash cannot hide detached active jobs from owned cleanup."""
    stub = tmp_path / "python_stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        """
import os,signal,sys,time
from pathlib import Path
if "-c" in sys.argv:
    if "latest_tree_snapshot_path" in sys.argv[-1]:print("fixture-source");sys.exit(0)
    sys.exit(1)
from chipiron.environments.morpion.bootstrap.persistent_worker_state import WorkerJobState
stage=sys.argv[sys.argv.index("--pipeline-stage")+1]
work=Path(os.environ["MORPION_WORK_DIR"])
state=WorkerJobState(work,stage);state.update("running",job_id=1)
signal.signal(signal.SIGTERM,lambda *_args:sys.exit(143))
(work/(stage+".ready")).write_text(str(os.getpid()))
while True:
    (work/(stage+".writer")).write_text(str(time.monotonic()))
    time.sleep(.01)
"""
    )
    stub.chmod(0o755)
    (tmp_path / "run_state.json").write_text(
        '{"latest_tree_snapshot_path":"fixture-source"}'
    )
    monkeypatch.setenv("MORPION_PERSISTENT_WORKERS", "1")
    original_popen = subprocess.Popen
    owned: list[int] = []
    foreign: list[subprocess.Popen[Any]] = []
    stages = ("growth", "dataset_worker", "training_worker", "reevaluation")

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        kwargs["env"] = dict(kwargs["env"], PYTHON_BIN=str(stub))
        process = original_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while not all((tmp_path / (stage + ".ready")).exists() for stage in stages):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        owned.extend(
            int((tmp_path / (stage + ".ready")).read_text()) for stage in stages
        )
        foreign_env = dict(kwargs["env"])
        if foreign_kind == "untagged":
            for key in (
                "MORPION_RESOURCE_MONITOR_PID",
                "MORPION_RESOURCE_MONITOR_START_TICKS",
                "MORPION_RESOURCE_MONITOR_SESSION",
                "MORPION_WORK_DIR",
            ):
                foreign_env.pop(key)
        elif foreign_kind == "ticks":
            foreign_env["MORPION_RESOURCE_MONITOR_START_TICKS"] += "0"
        elif foreign_kind == "work_dir":
            foreign_env["MORPION_WORK_DIR"] += "-other"
        else:
            foreign_env["MORPION_RESOURCE_MONITOR_SESSION"] += "-other"
        foreign.append(
            original_popen(
                [sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"],
                env=foreign_env,
                start_new_session=True,
            )
        )
        if foreign_kind == "forged_status":
            path = tmp_path / "pipeline/worker_processes/training_worker.json"
            record = json.loads(path.read_text())
            pid = foreign[0].pid
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            record["identity"].update(
                pid=pid, start_ticks=fields[19], session=int(fields[3])
            )
            path.write_text(json.dumps(record))
        shell = psutil.Process(process.pid).children()[0]
        os.kill(shell.pid, signal.SIGKILL)
        assert process.wait(timeout=3) == 137
        return process

    monkeypatch.setattr(operator.subprocess, "Popen", launch)
    monkeypatch.setattr(operator, "available_ram_mb", lambda: 20000.0)
    monkeypatch.setattr(
        operator,
        "load_bootstrap_run_state",
        lambda _p: SimpleNamespace(metadata={}, tree_size_at_last_save=1),
    )
    monitor = tmp_path / "resource_monitor"
    monitor.mkdir()
    try:
        with (
            ExitStack() as resources,
            pytest.raises(ValueError, match="Persistent worker group failed"),
        ):
            operator._run_monitored(
                MorpionBootstrapPaths.from_work_dir(tmp_path),
                Path(__file__).resolve().parents[2]
                / "scripts/launch_morpion_gnome_cluster.sh",
                MorpionContinualLearningPolicy(
                    max_elapsed_seconds=10, shutdown_grace_seconds=1
                ),
                monitor,
                psutil,
                resources,
            )
        survivors = [
            pid
            for pid in owned
            if psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        ]
        assert not survivors, f"Registered writers survived native cleanup: {survivors}"
        assert foreign[0].poll() is None, (
            "Cleanup must preserve the mismatched/untagged process"
        )
        receipt = next(
            json.loads(p.read_text())
            for p in monitor.glob("session_*.json")
            if not p.name.endswith(".native.json")
        )
        assert receipt["shutdown_confirmed"] and receipt["forced_group_cleanup"]
        assert not receipt["shutdown_grace_expired"]
        assert receipt["decision"]["supervisor_exit_code"] == 137
        assert set(owned) <= set(receipt["environment_owned_groups"])
    finally:
        for pid in owned:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        for child in foreign:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
