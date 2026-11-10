"""Persistent Python jobs keep bounded admission, lifetime and failure semantics."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import persistent_worker as worker
from chipiron.environments.morpion.bootstrap import persistent_worker_state as status
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    MorpionBootstrapLauncherArgs,
    launcher_args_from_cli,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.runtime_performance import (
    WorkerObservation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _identity() -> dict[str, Any]:
    """Use current process birth/session; actual ancestry has separate subprocess tests."""
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": os.getpid(),
        "start_ticks": fields[19],
        "session": int(fields[3]),
        "monitor_pid": os.getpid(),
        "monitor_start_ticks": fields[19],
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def _launcher(tmp_path: Path, **changes: Any) -> MorpionBootstrapLauncherArgs:
    """Construct an explicitly supported bounded operational recipe."""
    return replace(
        MorpionBootstrapLauncherArgs(
            MorpionBootstrapArgs(
                tmp_path,
                execution_profile="strict_math",
                torch_intra_op_threads=1,
                torch_inter_op_threads=1,
                training_device="cpu",
                continual_learning=MorpionContinualLearningPolicy(),
            ),
            scientific=True,
            pipeline_stage="training_worker",
            persistent_worker=True,
            worker_max_jobs=2,
            worker_poll_seconds=1,
        ),
        **changes,
    )


def _unmanaged_configuration(tmp_path: Path) -> None:
    """Keep dispatcher stubs valid at the real managed-enrollment guard."""
    save_bootstrap_config(
        bootstrap_config_from_args(_launcher(tmp_path).bootstrap_args),
        tmp_path / "bootstrap_config.json",
    )


@pytest.mark.fast
def test_default_cli_stays_fresh_and_optin_is_launcher_only() -> None:
    """Operational persistence does not add a scientific hyperparameter."""
    args = [
        "--work-dir",
        "/unused/private",
        "--scientific",
        "--pipeline-stage",
        "growth",
    ]
    before = launcher_args_from_cli(args)
    after = launcher_args_from_cli([
        *args,
        "--persistent-worker",
        "--worker-poll-seconds",
        "2",
        "--worker-max-jobs",
        "3",
    ])
    assert not before.persistent_worker and after.persistent_worker
    assert before.bootstrap_args == after.bootstrap_args
    assert after.worker_poll_seconds == 2 and after.worker_max_jobs == 3


@pytest.mark.fast
@pytest.mark.parametrize(
    "change",
    [
        {"scientific": False},
        {"pipeline_stage": "loop"},
        {"pipeline_stage": "dataset"},
        {"scientific_operation": "inspect"},
        {"pipeline_generation": 2},
        {"pipeline_stage": "growth", "max_cycles": 20},
        {"worker_poll_seconds": 0},
        {"worker_max_jobs": 0},
    ],
)
def test_unsupported_lifetime_refuses_before_dispatch(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """A persistent loop is never an alternate unbounded/manual stage scheduler."""
    with pytest.raises(ValueError):
        worker.run_persistent_worker(_launcher(tmp_path, **change))
    assert not list(tmp_path.iterdir())


@pytest.mark.fast
@pytest.mark.parametrize(
    "phase,age", [("running", 20.0), ("cleanup", 20.0), ("idle", 0.0)]
)
def test_monitor_separates_job_age_from_process_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    age: float,
) -> None:
    """Old idle Python is not an overlong job; cleanup remains part of its bound."""
    monkeypatch.setattr(status, "monitored_identity", _identity)
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(status.time, "monotonic", lambda: 100.0)
        state = status.WorkerJobState(tmp_path, "training_worker")
        try:
            state.update(phase, job_started_monotonic_s=80.0)
            assert status.active_job_age(tmp_path, os.getpid(), 1000.0) == age
        finally:
            state.close()


@pytest.mark.fast
@pytest.mark.parametrize(
    "damage",
    [
        "pid",
        "ticks",
        "session",
        "monitor",
        "namespace",
        "clock",
        "stale_idle",
        "missing",
        "corrupt",
    ],
)
def test_monitor_never_trusts_stale_or_foreign_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """Invalid status cannot hide a surviving old computation from the job limit."""
    monkeypatch.setattr(status, "monitored_identity", _identity)
    state = status.WorkerJobState(tmp_path, "training_worker")
    try:
        state.update("idle")
        d = json.loads(state.path.read_text())
        if damage == "pid":
            d["identity"]["pid"] += 1000000
        elif damage == "ticks":
            d["identity"]["start_ticks"] = "0"
        elif damage == "session":
            d["identity"]["session"] += 1
        elif damage == "monitor":
            d["identity"]["monitor_pid"] += 1
        elif damage == "namespace":
            d["identity"]["pid_namespace"] = "pid:[0]"
        elif damage == "clock":
            d["job_started_monotonic_s"] = float("nan")
        elif damage == "stale_idle":
            d["heartbeat_monotonic_s"] -= 6
        state.path.write_text(json.dumps(d))
        if damage == "missing":
            state.path.unlink()
        elif damage == "corrupt":
            state.path.write_text("{")
        assert status.active_job_age(tmp_path, os.getpid(), 1000.0) == 1000.0
    finally:
        state.close()


@pytest.mark.fast
def test_process_registration_excludes_second_live_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status of a live process cannot be overwritten by a second same-stage owner."""
    monkeypatch.setattr(status, "monitored_identity", _identity)
    first = status.WorkerJobState(tmp_path, "growth")
    try:
        before = first.path.read_bytes()
        with pytest.raises(BlockingIOError):
            status.WorkerJobState(tmp_path, "growth")
        assert first.path.read_bytes() == before
    finally:
        first.close()
    resumed = status.WorkerJobState(tmp_path, "growth")
    resumed.close()


@pytest.mark.fast
def test_job_observations_do_not_count_process_startup_repeatedly(
    tmp_path: Path,
) -> None:
    """Distinct jobs retain separate bounded records and one actual process startup."""
    process = WorkerObservation(tmp_path, "training_worker")
    for number in (1, 2):
        WorkerObservation(tmp_path, "training_worker", job_id=number).finish(0)
    process.finish(0)
    values = [
        json.loads(p.read_text())
        for p in (tmp_path / "pipeline/performance/runtime").glob("*.json")
    ]
    assert len([v for v in values if v["stage"] == "startup"]) == 1
    assert {v["job_id"] for v in values if v["stage"] == "worker_job"} == {1, 2}


@pytest.mark.parametrize("fail_at", [None, "job", "cleanup"])
def test_success_waits_only_after_cleanup_and_failure_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str | None,
) -> None:
    """Drive the real dispatcher with controlled work; no failed job becomes idle."""
    from chipiron.environments.morpion.bootstrap import launcher

    events: list[str] = []
    monkeypatch.setattr(
        worker, "monitored_identity", lambda: {**_identity(), "session": os.getpid()}
    )
    monkeypatch.setattr(status, "monitored_identity", _identity)
    _unmanaged_configuration(tmp_path)
    (tmp_path / "run_state.json").write_text("{}")

    @contextmanager
    def execution(
        args: MorpionBootstrapLauncherArgs,
    ) -> Iterator[MorpionBootstrapLauncherArgs]:
        events.append("resolve")
        yield args

    monkeypatch.setattr(worker, "launcher_execution", execution)

    def job(args: MorpionBootstrapLauncherArgs, *, _job_id: int | None = None) -> int:
        assert args.persistent_worker and _job_id == events.count("job") + 1
        events.append("job")
        if fail_at == "job":
            message = "injected job failure"
            raise ValueError(message)
        return 0

    monkeypatch.setattr(launcher, "_run_launcher_cli", job)

    def cleanup() -> None:
        events.append("cleanup")
        if fail_at == "cleanup":
            message = "injected cleanup failure"
            raise ValueError(message)

    monkeypatch.setattr(worker, "_cleanup", cleanup)
    if fail_at:
        with pytest.raises(ValueError, match="injected"):
            worker.run_persistent_worker(_launcher(tmp_path))
        assert events.count("job") == 1
    else:
        assert worker.run_persistent_worker(_launcher(tmp_path)) == 0
        assert events == ["resolve", "job", "cleanup", "resolve", "job", "cleanup"]
    record = json.loads(
        (tmp_path / "pipeline/worker_processes/training_worker.json").read_text()
    )
    assert record["phase"] == "stopped" and record["exit_code"] == int(
        fail_at is not None
    )
    assert record.get("completed_jobs", 0) == (2 if fail_at is None else 0)


def test_actual_monitor_ancestry_and_process_death_release_registration(
    tmp_path: Path,
) -> None:
    """A real child owns the record/lock until death; parent sees no orphan writer."""
    env = dict(
        os.environ,
        MORPION_RESOURCE_MONITOR_PID=str(os.getpid()),
        MORPION_RESOURCE_MONITOR_START_TICKS=_identity()["start_ticks"],
    )
    code = """import sys,time
from pathlib import Path
from chipiron.environments.morpion.bootstrap.persistent_worker_state import WorkerJobState
s=WorkerJobState(Path(sys.argv[1]),'training_worker');s.update('idle')
print('ready',flush=True);time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        assert status.active_job_age(tmp_path, child.pid, 1000.0) == 0.0
        child.kill()
        child.wait(timeout=5)
        assert status.active_job_age(tmp_path, child.pid, 1000.0) == 1000.0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if child.stdout is not None:
            child.stdout.close()


@pytest.mark.fast
@pytest.mark.parametrize("stage", ["dataset_worker", "reevaluation"])
def test_private_reader_scope_rotation_preserves_foreign_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Only existing same-source Reevaluation disk reuse survives a job boundary."""
    from types import SimpleNamespace

    source = SimpleNamespace(latest_tree_snapshot_path="generation-1")
    monkeypatch.setattr(worker, "load_bootstrap_run_state", lambda _: source)
    parent = tmp_path / "supervisor-cache"
    parent.mkdir()
    sentinel = parent / "foreign"
    sentinel.write_bytes(b"preserve")
    scope = worker._ReaderScope(
        _launcher(tmp_path, pipeline_stage=stage, reader_cache_dir=parent)
    )
    first = scope.next()
    assert first is not None
    marker = first / "index"
    marker.write_bytes(b"immutable private index")
    second = scope.next()
    assert second is not None
    if stage == "dataset_worker":
        assert second != first and not first.exists()
    else:
        assert second == first and marker.read_bytes() == b"immutable private index"
    source.latest_tree_snapshot_path = "generation-2"
    third = scope.next()
    assert third is not None and third != second and not second.exists()
    scope.close()
    assert not third.exists() and sentinel.read_bytes() == b"preserve"


def test_cleanup_refuses_orphan_in_owned_session(tmp_path: Path) -> None:
    """A reparented grandchild is still work in the session, never safe idle."""
    import signal

    pid_file = tmp_path / "orphan.pid"
    code = """import subprocess,sys,time
from pathlib import Path
from chipiron.environments.morpion.bootstrap.persistent_worker import _cleanup
child=subprocess.Popen([sys.executable,'-c',"import os,time; from pathlib import Path; pid=os.fork(); (Path(__import__('sys').argv[1]).write_text(str(os.getpid())),time.sleep(40)) if pid==0 else None",sys.argv[1]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
child.wait(timeout=5)
for _ in range(100):
    if Path(sys.argv[1]).is_file():break
    time.sleep(.02)
try:_cleanup()
except ValueError as e:
    assert 'left child processes' in str(e); print('refused',flush=True)
else:raise AssertionError('orphan hidden by reparenting')
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(pid_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=25)
        assert process.returncode == 0 and "refused" in output, output
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    assert pid_file.is_file()
    orphan = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            phase = Path(f"/proc/{orphan}/stat").read_text().rsplit(") ", 1)[1][0]
        except FileNotFoundError:
            break
        if phase in {"Z", "X"}:
            break
        time.sleep(0.02)
    else:
        pytest.fail("orphan still executing after owned group kill")


@pytest.mark.parametrize("number", [2, 15])
def test_orderly_signal_status_preserves_actual_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    """A normal interrupted worker is stopped in K1, never a recorded exit1 failure."""
    from chipiron.environments.morpion.bootstrap import launcher

    monkeypatch.setattr(
        worker, "monitored_identity", lambda: {**_identity(), "session": os.getpid()}
    )
    monkeypatch.setattr(status, "monitored_identity", _identity)
    _unmanaged_configuration(tmp_path)
    (tmp_path / "run_state.json").write_text("{}")

    @contextmanager
    def execution(
        args: MorpionBootstrapLauncherArgs,
    ) -> Iterator[MorpionBootstrapLauncherArgs]:
        yield args

    monkeypatch.setattr(worker, "launcher_execution", execution)

    def interrupt(*_args: Any, **_kwargs: Any) -> int:
        worker._interrupt(number, None)
        message = "Signal must exit"
        raise AssertionError(message)

    monkeypatch.setattr(launcher, "_run_launcher_cli", interrupt)
    with pytest.raises(SystemExit) as caught:
        worker.run_persistent_worker(_launcher(tmp_path))
    assert caught.value.code == 128 + number
    record = json.loads(
        (tmp_path / "pipeline/worker_processes/training_worker.json").read_text()
    )
    assert record["phase"] == "stopped" and record["exit_code"] == caught.value.code
    assert record.get("completed_jobs", 0) == 0


@pytest.mark.parametrize("contention", [True, False])
def test_growth_lease_contention_polls_but_computation_errors_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contention: bool
) -> None:
    """Retry only an actual unavailable lease; never hide arbitrary BlockingIOError."""
    import fcntl

    from chipiron.environments.morpion.bootstrap import launcher
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        growth_receipt_ownership,
    )

    monkeypatch.setattr(
        worker, "monitored_identity", lambda: {**_identity(), "session": os.getpid()}
    )
    monkeypatch.setattr(status, "monitored_identity", _identity)
    _unmanaged_configuration(tmp_path)
    (tmp_path / "run_state.json").write_text("{}")
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.pipeline_dir.mkdir()
    lease = (paths.pipeline_dir / ".growth_patch_consumer.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls: list[int | None] = []

    @contextmanager
    def execution(
        args: MorpionBootstrapLauncherArgs,
    ) -> Iterator[MorpionBootstrapLauncherArgs]:
        yield args

    def job(_args: MorpionBootstrapLauncherArgs, *, _job_id: int | None = None) -> int:
        calls.append(_job_id)
        if not contention:
            message = "unrelated computation failed"
            raise BlockingIOError(message)
        with growth_receipt_ownership(paths):
            assert _job_id == 2
        return 0

    monkeypatch.setattr(worker, "launcher_execution", execution)
    monkeypatch.setattr(launcher, "_run_launcher_cli", job)
    monkeypatch.setattr(worker, "_cleanup", lambda: fcntl.flock(lease, fcntl.LOCK_UN))
    try:
        if contention:
            assert (
                worker.run_persistent_worker(
                    _launcher(tmp_path, pipeline_stage="growth")
                )
                == 0
            )
            assert calls == [1, 2]
        else:
            with pytest.raises(BlockingIOError, match="unrelated computation failed"):
                worker.run_persistent_worker(
                    _launcher(tmp_path, pipeline_stage="growth")
                )
            assert calls == [1]
    finally:
        lease.close()
