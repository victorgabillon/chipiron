"""Deterministic monitor read races, without scientific jobs or long waits."""

import json
import os
import signal
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import continual_operator as operator
from chipiron.environments.morpion.bootstrap import persistent_worker_state as status
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from tests.environments.test_morpion_persistent_workers import _identity


def run_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    before: float = 2199.9,
    after: float = 2200.2,
    phase: str = "stopped",
    job_start: float = 2190.0,
    termination: str | None = None,
    interrupt: bool = False,
    wall_jump: float = 0.0,
    disappear: bool = False,
    disappear_at_stat: bool = False,
    fail_shutdown: bool = False,
    initial_delay: float = 0.0,
    consumed: float = 0.0,
    after_state: float | None = None,
) -> dict[str, Any]:
    """Drive the actual monitor loop; sampling/cleanup advance controlled clocks."""
    clock = [1000.0]
    monitor = tmp_path / "resource_monitor"
    monitor.mkdir()
    if consumed:
        (monitor / "usage.json").write_text(json.dumps({"elapsed_seconds": consumed}))
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    monkeypatch.setattr(status, "monitored_identity", _identity)
    monkeypatch.setattr(operator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(operator.time, "time", lambda: 10000.0 + wall_jump)
    ticks = int(_identity()["start_ticks"]) / os.sysconf("SC_CLK_TCK")
    monkeypatch.setattr(
        operator.time, "clock_gettime", lambda _kind: ticks + clock[0] - 1005.0
    )
    state = status.WorkerJobState(tmp_path, "growth")
    state.update(phase, job_started_monotonic_s=job_start)
    record = state.record | {"heartbeat_monotonic_s": before}
    state.path.write_text(json.dumps(record))
    state.close()
    original_stat = Path.read_text

    def proc_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if disappear_at_stat and path == Path(f"/proc/{os.getpid()}/stat"):
            raise FileNotFoundError
        return original_stat(path, *args, **kwargs)

    class GoneError(Exception):
        """Fake psutil process-disappearance exception."""

    class Child:
        pid = os.getpid()

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1024)

        def cmdline(self) -> list[str]:
            if disappear:
                raise GoneError
            return ["python", "--pipeline-stage", "growth", "--persistent-worker"]

        def create_time(self) -> float:
            return 8805.0

        def children(self, *, recursive: bool = False) -> list[Any]:
            return []

    class Supervisor:
        pid = os.getpid()
        returncode: int | None = None
        started = False
        limit_seconds: int | None = None

        def __enter__(self) -> Supervisor:
            self.started = True
            clock[0] = before
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            assert timeout in (11, 5)
            if fail_shutdown and timeout != 5:
                raise operator.subprocess.TimeoutExpired("dummy", timeout)
            # A late cleanup must never relabel an earlier real job failure.
            clock[0] = 2205.0
            self.returncode = 0
            return 0

    supervisor = Supervisor()

    def launch(*_args: Any, **kwargs: Any) -> Supervisor:
        supervisor.limit_seconds = int(kwargs["env"]["MORPION_CLUSTER_MAX_SECONDS"])
        return supervisor

    monkeypatch.setattr(operator.subprocess, "Popen", launch)
    monkeypatch.setattr(Path, "read_text", proc_read)
    monkeypatch.setattr(operator, "available_ram_mb", lambda: 20000.0)
    monkeypatch.setattr(
        operator.shutil, "disk_usage", lambda _path: SimpleNamespace(free=50000 * 2**20)
    )

    def disk(_path: Path) -> dict[str, int]:
        if not supervisor.started:
            clock[0] = 1000.0 + initial_delay
        else:
            clock[0] = after
            if interrupt:
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
        return {"allocated_bytes": 0}

    monkeypatch.setattr(operator, "directory_disk_usage", disk)
    monkeypatch.setattr(
        operator.time, "sleep", lambda _seconds: setattr(supervisor, "returncode", 0)
    )
    authority = SimpleNamespace(
        metadata={}
        if termination is None
        else {"experiment_termination": {"reason": termination}},
        tree_size_at_last_save=1,
    )

    def load_state(_path: Path) -> SimpleNamespace:
        if after_state is not None:
            clock[0] = after_state
        return authority

    monkeypatch.setattr(operator, "load_bootstrap_run_state", load_state)
    monkeypatch.setattr(operator, "publication_records", lambda _meta: [])
    usages: list[dict[str, Any]] = []
    original_atomic = operator.atomic_json

    def persist(path: Path, value: dict[str, Any]) -> None:
        if path == monitor / "usage.json":
            usages.append(dict(value))
        original_atomic(path, value)

    monkeypatch.setattr(operator, "atomic_json", persist)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        operator.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    process_module = SimpleNamespace(
        Process=lambda _pid: Child(),
        NoSuchProcess=GoneError,
        process_iter=lambda _fields: [],
        STATUS_ZOMBIE="zombie",
        STATUS_DEAD="dead",
    )
    policy = MorpionContinualLearningPolicy(
        max_elapsed_seconds=1200, worker_max_seconds=900, shutdown_grace_seconds=1
    )
    with ExitStack() as resources:
        operator._run_monitored(  # pylint: disable=protected-access
            paths,
            tmp_path / "unused-supervisor",
            policy,
            monitor,
            process_module,
            resources,
        )
    receipt = json.loads(next(monitor.glob("session_*.json")).read_text())
    receipt["decision"] = usages[0]
    receipt["forced_signals"] = signals
    receipt["supervisor_started"] = supervisor.started
    receipt["supervisor_limit_seconds"] = supervisor.limit_seconds
    assert receipt["shutdown_confirmed"]
    assert not Path(receipt["scratch"]).exists()
    return receipt


@pytest.mark.fast
def test_sampling_crosses_deadline_before_stopped_record_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-deadline sample cannot outrank the actual post-sampling deadline."""
    result = run_sample(tmp_path, monkeypatch)
    assert result["reason"] == "maximum_elapsed_time"
    assert result["decision"]["elapsed_seconds"] == pytest.approx(1200.2)


@pytest.mark.fast
def test_state_read_also_precedes_decision_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading committed state is part of the potentially slow sample boundary."""
    result = run_sample(tmp_path, monkeypatch, after=2199.95, after_state=2200.2)
    assert result["reason"] == "maximum_elapsed_time"
    assert result["decision"]["elapsed_seconds"] == pytest.approx(1200.2)


@pytest.mark.fast
@pytest.mark.parametrize("phase", ["running", "cleanup", "idle"])
def test_record_written_during_read_is_not_a_future_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Atomic publication can be newer than the clock taken before reading it."""
    clock = [100.0]
    monkeypatch.setattr(status, "monitored_identity", _identity)
    monkeypatch.setattr(status.time, "monotonic", lambda: clock[0])
    state = status.WorkerJobState(tmp_path, "training_worker")
    state.update(phase, job_started_monotonic_s=99.0)
    original_open = Path.open

    def concurrent_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == state.path and args == ("rb",):
            clock[0] = 100.1
            with original_open(path, "w") as stream:
                json.dump(state.record | {"heartbeat_monotonic_s": 100.1}, stream)
        return original_open(path, *args, **kwargs)

    try:
        monkeypatch.setattr(Path, "open", concurrent_open)
        assert status.active_job_age(tmp_path, os.getpid(), 1195.0) == pytest.approx(
            0.0 if phase == "idle" else 1.1
        )
    finally:
        state.close()


@pytest.mark.fast
@pytest.mark.parametrize("wall_jump", [-100000.0, 100000.0])
def test_wall_clock_change_does_not_invent_unknown_status_process_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wall_jump: float
) -> None:
    """Conservative stopped fallback uses kernel process lifetime, not Unix jumps."""
    result = run_sample(
        tmp_path,
        monkeypatch,
        before=1100.0,
        after=1100.1,
        job_start=1090.0,
        wall_jump=wall_jump,
    )
    assert result["reason"] == "supervisor_exit"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("phase", "job_start", "expected"),
    [
        ("running", 2140.0, "supervisor_exit"),
        ("running", 1000.0, "maximum_worker_runtime"),
        ("cleanup", 1000.0, "maximum_worker_runtime"),
        ("stopped", 2140.0, "maximum_worker_runtime"),
    ],
)
def test_early_overrun_is_not_relabelled_after_cleanup_crosses_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    job_start: float,
    expected: str,
) -> None:
    """Healthy short jobs survive; true overrun and stuck stopped children do not."""
    result = run_sample(
        tmp_path,
        monkeypatch,
        before=2150.0,
        after=2150.1,
        phase=phase,
        job_start=job_start,
    )
    assert result["reason"] == expected
    assert result["budget_elapsed_seconds"] >= 1150
    assert result["decision"]["elapsed_seconds"] == pytest.approx(1150.1)


@pytest.mark.fast
@pytest.mark.parametrize("interrupt", [False, True])
def test_authority_and_operator_precede_resource_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, interrupt: bool
) -> None:
    """No accounting correction overrides authoritative/operator termination."""
    result = run_sample(
        tmp_path, monkeypatch, termination="scientific_failure", interrupt=interrupt
    )
    assert result["reason"] == (
        "operator_SIGINT" if interrupt else "scientific_failure"
    )


@pytest.mark.fast
def test_disappearing_process_does_not_create_worker_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child reaped during sampling is omitted, not reported as a long job."""
    result = run_sample(
        tmp_path, monkeypatch, before=2150.0, after=2150.1, disappear=True
    )
    assert result["reason"] == "supervisor_exit"


@pytest.mark.fast
def test_process_disappears_at_kernel_birth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Procfs disappearance between process fields is the same reaping race."""
    result = run_sample(
        tmp_path, monkeypatch, before=2150.0, after=2150.1, disappear_at_stat=True
    )
    assert result["reason"] == "supervisor_exit"


@pytest.mark.fast
def test_initial_sample_spends_remaining_budget_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted budget cannot start another supervisor after slow preflight."""
    result = run_sample(tmp_path, monkeypatch, consumed=1199.0, initial_delay=2.0)
    assert result["reason"] == "maximum_elapsed_time"
    assert not result["supervisor_started"]


@pytest.mark.fast
def test_supervisor_budget_is_not_rounded_down_by_sample_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subsecond sampling must not shorten the existing integer supervisor cap."""
    result = run_sample(tmp_path, monkeypatch, initial_delay=0.01)
    assert result["supervisor_limit_seconds"] == 1200


@pytest.mark.fast
@pytest.mark.parametrize("wall_jump", [-100000.0, 100000.0])
def test_wall_clock_cannot_hide_true_fallback_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wall_jump: float
) -> None:
    """An untrusted stopped record cannot hide an old process after a Unix jump."""
    result = run_sample(
        tmp_path,
        monkeypatch,
        before=2150.0,
        after=2150.1,
        job_start=2140.0,
        wall_jump=wall_jump,
    )
    assert result["reason"] == "maximum_worker_runtime"


@pytest.mark.fast
@pytest.mark.parametrize(
    "damage",
    [
        "boot",
        "monitor_ticks",
        "future_heartbeat",
        "future_job",
        "boolean_clock",
        "malformed",
        "oversize",
        "stopped",
        "missing_proc",
        "stale_idle",
    ],
)
def test_post_read_clock_does_not_trust_bad_or_stopped_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Post-read freshness keeps the existing identity/clock/idle bounds strict."""
    monkeypatch.setattr(status, "monitored_identity", _identity)
    monkeypatch.setattr(status.time, "monotonic", lambda: 100.0)
    state = status.WorkerJobState(tmp_path, "training_worker")
    try:
        state.update("idle", job_started_monotonic_s=90.0)
        data = dict(state.record)
        if damage == "boot":
            data["identity"]["boot"] = "other-boot"
        elif damage == "monitor_ticks":
            data["identity"]["monitor_start_ticks"] = "0"
        elif damage == "future_heartbeat":
            data["heartbeat_monotonic_s"] = 100.01
        elif damage == "future_job":
            data["job_started_monotonic_s"] = 100.01
        elif damage == "boolean_clock":
            data["heartbeat_monotonic_s"] = True
        elif damage == "stale_idle":
            data["heartbeat_monotonic_s"] = 94.99
        elif damage == "stopped":
            data["phase"] = "stopped"
        elif damage == "oversize":
            data["padding"] = "x" * status.MAX_STATUS_BYTES
        state.path.write_text("{" if damage == "malformed" else json.dumps(data))
        pid = 999999999 if damage == "missing_proc" else os.getpid()
        assert status.active_job_age(tmp_path, pid, 1195.0) == 1195.0
    finally:
        state.close()


@pytest.mark.fast
def test_cleanup_remains_bounded_after_accounting_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing cleanup grace then group kill remains intact."""
    result = run_sample(tmp_path, monkeypatch, fail_shutdown=True)
    assert result["reason"] == "maximum_elapsed_time"
    assert result["forced_signals"] == [(os.getpgid(os.getpid()), signal.SIGKILL)]
