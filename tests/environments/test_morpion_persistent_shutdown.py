"""Owned process shutdown accounting without scientific work or a long wait."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import psutil
import pytest

from chipiron.environments.morpion.bootstrap import continual_operator as operator
from chipiron.environments.morpion.bootstrap import persistent_worker_state as status
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _owned_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    *,
    phase: str = "stopped",
    job_age: float = 183.0,
    elapsed: float = 7199.8,
    interrupt: bool = False,
    cross_decision_clock: bool = False,
    control_mode: str = "normal",
    cleanup_elapsed: float = 7201.5,
    exit_before_cleanup: bool = False,
) -> dict[str, Any]:
    """Only the native age clocks are advanced; every process identity is real."""
    original_popen = subprocess.Popen
    launched: list[Any] = []
    real_sleep = time.sleep
    real_monotonic = time.monotonic
    base = real_monotonic()
    clock = [base]
    monitor = tmp_path / "resource_monitor"
    monitor.mkdir()
    supervisor = tmp_path / "supervisor.sh"
    # An inert process supplies real worker birth/session/ancestry. No launcher,
    # Training, Growth, neural inference or scientific checkpoint is invoked.
    supervisor.write_text("""#!/usr/bin/env bash
set -eu
trap 'kill -TERM -- "-$worker" 2>/dev/null || true; wait "$worker" 2>/dev/null || true; exit 0' TERM
(
  if [[ -n "${MORPION_SUPERVISOR_SHUTDOWN_FD:-}" ]]; then exec {MORPION_SUPERVISOR_SHUTDOWN_FD}>&-; fi
  exec setsid "$PYTHON_BIN" -I -B -c 'import time; time.sleep(30)' --pipeline-stage training_worker --persistent-worker
) &
worker=$!
export TEST_WORKER_PID=$worker
"$PYTHON_BIN" -I -B -c '
import os,json,time
from pathlib import Path
root=Path(os.environ["MORPION_WORK_DIR"]); now=float(os.environ["TEST_NOW"])
def fields(pid):return Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()
pid=int(os.environ["TEST_WORKER_PID"]); wf=fields(pid); sp=os.getppid(); sf=fields(sp)
common={"monitor_pid":int(os.environ["MORPION_RESOURCE_MONITOR_PID"]),"monitor_start_ticks":os.environ["MORPION_RESOURCE_MONITOR_START_TICKS"],"boot":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"pid_namespace":os.readlink("/proc/self/ns/pid")}
intent=common|{"schema_version":1,"monitor_session":os.environ.get("MORPION_RESOURCE_MONITOR_SESSION","baseline-fixture"),"supervisor_pid":sp,"supervisor_start_ticks":sf[19],"supervisor_session":int(sf[3]),"reason":"supervisor_deadline","requested_monotonic_s":now-.5,"limit_seconds":int(os.environ["MORPION_CLUSTER_MAX_SECONDS"]),"grace_seconds":int(os.environ["MORPION_CLUSTER_STOP_GRACE_SECONDS"])}
if "MORPION_SUPERVISOR_SHUTDOWN_FD" in os.environ:os.write(int(os.environ["MORPION_SUPERVISOR_SHUTDOWN_FD"]),json.dumps(intent).encode()+b"\\n")
(root/"fixture-intent.json").write_text(json.dumps(intent))
identity=common|{"pid":pid,"start_ticks":wf[19],"session":int(wf[3])}
path=root/"pipeline/worker_processes/training_worker.json";path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({"schema_version":1,"worker":"training_worker","identity":identity,"phase":os.environ["TEST_PHASE"],"heartbeat_monotonic_s":now-.1,"job_started_monotonic_s":now-float(os.environ["TEST_JOB_AGE"]),"exit_code":int(os.environ["TEST_EXIT"])}))
(root/"ready").write_text(str(pid))
'
wait "$worker" || true
""")

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        kwargs["env"] = dict(
            kwargs["env"],
            TEST_NOW=str(base + elapsed),
            TEST_EXIT=str(exit_code),
            TEST_PHASE=phase,
            TEST_JOB_AGE=str(job_age),
        )
        handle = original_popen(*args, **kwargs)
        launched.append(handle)
        deadline = real_monotonic() + 5
        while not (tmp_path / "ready").exists() and handle.poll() is None:
            assert real_monotonic() < deadline
            real_sleep(0.01)
        assert (tmp_path / "ready").exists()
        clock[0] = base + elapsed
        if interrupt:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
        original_wait = handle.wait

        def wait(*args: Any, **kwargs: Any) -> int:
            result = original_wait(*args, **kwargs)
            clock[0] = base + cleanup_elapsed  # Separate from the decision clock.
            return result

        monkeypatch.setattr(handle, "wait", wait)
        return handle

    def boottime(_kind: int) -> float:
        pid = int((tmp_path / "ready").read_text())
        ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
        return int(ticks) / os.sysconf("SC_CLK_TCK") + 7199.5

    monkeypatch.setattr(operator.subprocess, "Popen", launch)
    if exit_before_cleanup:
        original_atomic = operator.atomic_json

        def persist_after_exit(path: Path, value: dict[str, Any]) -> None:
            """Make the real owned supervisor exit before final cleanup polling."""
            original_atomic(path, value)
            if path.name.startswith("decision_"):
                handle = launched[0]
                for child in psutil.Process(handle.pid).children():
                    os.kill(child.pid, signal.SIGTERM)
                handle.wait(timeout=3)

        monkeypatch.setattr(operator, "atomic_json", persist_after_exit)
    original_pair = operator.socket.socketpair

    class FragmentedSocket:
        """Keep the real channel but control its stream framing at the reader."""

        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.seen = False

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def recv(self, _size: int) -> bytes:
            if control_mode == "partial_eof" and self.seen:
                return b""
            value = self.wrapped.recv(13 if control_mode != "duplicate" else _size)
            self.seen = bool(value)
            return value + b"{}\n" if value and control_mode == "duplicate" else value

    def socketpair() -> tuple[Any, Any]:
        parent, child = original_pair()
        return FragmentedSocket(parent), child

    if control_mode != "normal":
        monkeypatch.setattr(operator.socket, "socketpair", socketpair)
    monkeypatch.setattr(operator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(operator.time, "clock_gettime", boottime)
    monkeypatch.setattr(operator, "available_ram_mb", lambda: 20000.0)
    authority = SimpleNamespace(metadata={}, tree_size_at_last_save=1)

    def load_state(_path: Path) -> SimpleNamespace:
        if cross_decision_clock:
            clocks = iter([base + elapsed])
            monkeypatch.setattr(
                operator.time,
                "monotonic",
                lambda: next(clocks, base + 7200.1),
            )
        return authority

    monkeypatch.setattr(operator, "load_bootstrap_run_state", load_state)
    monkeypatch.setattr(operator, "publication_records", lambda _meta: [])
    group_failure = False
    with ExitStack() as resources:
        try:
            operator._run_monitored(  # pylint: disable=protected-access
                MorpionBootstrapPaths.from_work_dir(tmp_path),
                supervisor,
                MorpionContinualLearningPolicy(
                    max_elapsed_seconds=7200,
                    worker_max_seconds=1800,
                    shutdown_grace_seconds=5,
                ),
                monitor,
                psutil,
                resources,
            )
        except ValueError as exc:
            assert "Persistent worker group failed" in str(exc)
            group_failure = True
    receipt = next(
        value
        for path in monitor.glob("session_*.json")
        if "policy" in (value := json.loads(path.read_text()))
    )
    pid = int((tmp_path / "ready").read_text())
    assert not psutil.pid_exists(pid)
    receipt["test_group_failure"] = group_failure
    return receipt


@pytest.mark.fast
@pytest.mark.parametrize("exit_code", [0, 143])
def test_expected_cluster_deadline_shutdown_does_not_charge_persistent_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """A supervisor-owned stop before native7200 is not a7199s Training job."""
    result = _owned_deadline(tmp_path, monkeypatch, exit_code)
    assert result["reason"] == "supervisor_deadline"
    assert result["budget_elapsed_seconds"] > 7200
    assert result["decision"]["budget_elapsed_seconds"] == pytest.approx(7199.8)
    assert result["decision"]["reason"] == "supervisor_deadline"
    assert result["shutdown_confirmed"]
    assert not result["test_group_failure"]


@pytest.mark.fast
def test_decision_uses_one_clock_for_elapsed_and_job_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second clock read cannot invent a worker overrun across native7200."""
    result = _owned_deadline(
        tmp_path,
        monkeypatch,
        0,
        phase="cleanup",
        job_age=1799.9,
        cross_decision_clock=True,
    )
    decision = result["decision"]
    assert decision["reason"] == "supervisor_deadline"
    assert decision["budget_elapsed_seconds"] == pytest.approx(7199.8)
    assert decision["workers"][0]["accounted_job_age_s"] == pytest.approx(1799.9)
    assert decision["sample"]["longest_worker_seconds"] == pytest.approx(1799.9)


@pytest.mark.fast
@pytest.mark.parametrize(
    "phase,age,elapsed,interrupt,exit_code,expected",
    [
        ("running", 1799.0, 7199.8, False, 0, "supervisor_deadline"),
        ("running", 1801.0, 7199.8, False, 0, "maximum_worker_runtime"),
        ("cleanup", 1799.0, 7199.8, False, 0, "supervisor_deadline"),
        ("cleanup", 1801.0, 7199.8, False, 0, "maximum_worker_runtime"),
        ("stopped", 183.0, 7199.8, False, 1, "persistent_worker_failed"),
        ("stopped", 183.0, 7200.0, False, 143, "maximum_elapsed_time"),
        ("stopped", 183.0, 7200.0, False, 1, "maximum_elapsed_time"),
        ("cleanup", 1801.0, 7200.0, False, 0, "maximum_elapsed_time"),
        ("cleanup", 1801.0, 7200.0, True, 0, "operator_SIGINT"),
        ("stopped", 183.0, 7200.0, True, 1, "operator_SIGINT"),
    ],
)
def test_owned_monitor_preserves_deadline_failure_and_cancellation_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    age: float,
    elapsed: float,
    interrupt: bool,
    exit_code: int,
    expected: str,
) -> None:
    """Exercise the actual decision loop with trusted intent and genuine process identity."""
    receipt = _owned_deadline(
        tmp_path,
        monkeypatch,
        exit_code,
        phase=phase,
        job_age=age,
        elapsed=elapsed,
        interrupt=interrupt,
    )
    assert receipt["reason"] == expected
    assert receipt["decision"]["reason"] == expected
    assert receipt["decision"]["budget_elapsed_seconds"] == pytest.approx(elapsed)
    assert receipt["test_group_failure"] == (exit_code == 1)


@pytest.mark.fast
@pytest.mark.parametrize("mode", ["fragmented", "partial_eof", "duplicate"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_owned_control_stream_framing_and_failure_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    interrupt: bool,
) -> None:
    """Fragmented records work; closed partial/duplicate records fail even on cancel."""
    receipt = _owned_deadline(
        tmp_path, monkeypatch, 143, control_mode=mode, interrupt=interrupt
    )
    invalid = mode != "fragmented"
    assert receipt["reason"] == (
        "operator_SIGINT"
        if interrupt
        else "invalid_supervisor_shutdown"
        if invalid
        else "supervisor_deadline"
    )
    assert receipt["test_group_failure"] == invalid
    assert (receipt["decision"]["control_failure"] is not None) == invalid


@pytest.mark.fast
@pytest.mark.parametrize("already_exited", [False, True])
def test_ack_wait_counts_towards_absolute_shutdown_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_exited: bool
) -> None:
    """Even an exit0 cannot turn completion after declared grace into healthy shutdown."""
    receipt = _owned_deadline(
        tmp_path,
        monkeypatch,
        0,
        cleanup_elapsed=7205,
        exit_before_cleanup=already_exited,
    )
    assert receipt["reason"] == "supervisor_shutdown_grace_expired"
    assert receipt["decision"]["reason"] == "supervisor_deadline"
    assert receipt["shutdown_grace_expired"] and receipt["test_group_failure"]


@pytest.fixture
def owned_identity(tmp_path: Path) -> Iterator[tuple[int, int]]:
    """Keep a real two-level process/session alive for kernel identity checks."""
    info = tmp_path / "owned.json"
    code = (
        "import subprocess,time,json; from pathlib import Path; "
        "p=subprocess.Popen(['sleep','30']); "
        f"Path({str(info)!r}).write_text(json.dumps(p.pid)); p.wait()"
    )
    owner = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", code], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 3
        while not info.exists():
            assert owner.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        yield owner.pid, int(info.read_text())
    finally:
        os.killpg(owner.pid, signal.SIGTERM)
        owner.wait(timeout=3)


def _intent_record(parent: int, pid: int, now: float) -> dict[str, Any]:
    """Produce the operational wire fields from actual owned kernel identities."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    monitor = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    assert int(fields[1]) == parent
    return {
        "schema_version": 1,
        "monitor_session": "owned-test",
        "monitor_pid": os.getpid(),
        "monitor_start_ticks": monitor[19],
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "supervisor_pid": pid,
        "supervisor_start_ticks": fields[19],
        "supervisor_session": int(fields[3]),
        "reason": "supervisor_deadline",
        "requested_monotonic_s": now - 0.5,
        "limit_seconds": 7200,
        "grace_seconds": 60,
    }


def _validate(
    record: dict[str, Any], parent: int, now: float
) -> status.SupervisorShutdown:
    """Exercise validation at the same monitor-owned boundary as the socket consumer."""
    return status.validate_shutdown_intent(
        record,
        monitor_session="owned-test",
        parent_pid=parent,
        limit_seconds=7200,
        grace_seconds=60,
        not_before=now - 1,
        now=now,
    )


def _worker_record(parent: int, pid: int, now: float) -> dict[str, Any]:
    """Build a real live matching worker record with a controlled last job clock."""
    identity = _intent_record(parent, pid, now)
    return {
        "schema_version": 1,
        "worker": "training_worker",
        "identity": {
            k: identity[k]
            for k in ["monitor_pid", "monitor_start_ticks", "boot", "pid_namespace"]
        }
        | {
            "pid": pid,
            "start_ticks": identity["supervisor_start_ticks"],
            "session": identity["supervisor_session"],
        },
        "phase": "stopped",
        "exit_code": 143,
        "heartbeat_monotonic_s": now - 0.1,
        "job_started_monotonic_s": now - 183,
    }


def _reading(
    tmp_path: Path, pid: int, record: dict[str, Any]
) -> status.WorkerJobStatus:
    """Read bounded telemetry through the actual production status parser."""
    path = tmp_path / "pipeline/worker_processes/training_worker.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return status.read_job_status(tmp_path, pid, 7199.5)


@pytest.mark.fast
@pytest.mark.parametrize("phase", ["starting", "running", "cleanup", "stopped"])
@pytest.mark.parametrize("age", [1799.0, 1801.0])
def test_real_job_bound_survives_authorized_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_identity: tuple[int, int],
    phase: str,
    age: float,
) -> None:
    """Intent does not exempt running/cleanup or a stopped job that truly overran."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    monkeypatch.setattr(status.time, "monotonic", lambda: now)
    intent = _validate(_intent_record(parent, pid, now), parent, now)
    record = _worker_record(parent, pid, now)
    record.update(
        phase=phase, heartbeat_monotonic_s=now, job_started_monotonic_s=now - age
    )
    reading = _reading(tmp_path, pid, record)
    assert reading.active_age(intent, now) == age
    sample = dict(
        available_ram_mb=30000.0,
        free_disk_mb=50000.0,
        group_rss_mb=1.0,
        workspace_mb=1.0,
        scratch_mb=1.0,
        longest_worker_seconds=reading.active_age(intent, now),
    )
    policy = MorpionContinualLearningPolicy(
        max_elapsed_seconds=7200, worker_max_seconds=1800
    )
    assert operator.resource_limit_reason(policy, sample, 7199.8) == (
        "maximum_worker_runtime" if age > 1800 else None
    )
    assert (
        operator.resource_limit_reason(policy, sample, 7200) == "maximum_elapsed_time"
    )


@pytest.mark.fast
@pytest.mark.parametrize("code", [0, 143, 1, True, None])
def test_stopped_requires_authorized_context_and_successful_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_identity: tuple[int, int],
    code: int | None,
) -> None:
    """A stopped record and SIGTERM code alone never authorize benign shutdown."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    monkeypatch.setattr(status.time, "monotonic", lambda: now)
    intent = _validate(_intent_record(parent, pid, now), parent, now)
    record = _worker_record(parent, pid, now)
    record["exit_code"] = code
    reading = _reading(tmp_path, pid, record)
    if type(code) is int and code in {0, 143}:
        assert reading.active_age(intent, now) == pytest.approx(182.9)
        assert (
            reading.shutdown_failure(None, now) == "unexpected_persistent_worker_stop"
        )
        assert (
            reading.shutdown_failure(intent, now + 61)
            == "supervisor_shutdown_grace_expired"
        )
    else:
        assert reading.shutdown_failure(intent, now) == "persistent_worker_failed"
    assert reading.active_age(None, now) == 7199.5
    assert status.active_job_age(tmp_path, pid, 7199.5) == 7199.5


@pytest.mark.fast
@pytest.mark.parametrize(
    "field,value",
    [
        ("monitor_session", "other"),
        ("schema_version", True),
        ("monitor_pid", -1),
        ("monitor_start_ticks", "0"),
        ("boot", "other"),
        ("pid_namespace", "pid:[0]"),
        ("supervisor_pid", True),
        ("supervisor_start_ticks", "0"),
        ("supervisor_session", -1),
        ("reason", "cancelled"),
        ("limit_seconds", 7201),
        ("limit_seconds", True),
        ("grace_seconds", 61),
        ("grace_seconds", True),
        ("requested_monotonic_s", True),
        ("requested_monotonic_s", float("nan")),
        ("requested_monotonic_s", float("inf")),
        ("requested_monotonic_s", -1),
        ("requested_monotonic_s", "invalid"),
        ("requested_monotonic_s", None),
    ],
)
def test_stale_forged_or_mismatched_shutdown_intent_is_rejected(
    owned_identity: tuple[int, int],
    field: str,
    value: object,
) -> None:
    """Operational intent must bind this actual supervisor/monitor/session/budget."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    record = _intent_record(parent, pid, now)
    record[field] = value
    with pytest.raises((ValueError, OSError)):
        _validate(record, parent, now)


@pytest.mark.fast
@pytest.mark.parametrize("offset", [-61.0, -2.0, 1.0])
def test_shutdown_intent_clock_must_match_declared_deadline(
    owned_identity: tuple[int, int],
    offset: float,
) -> None:
    """Stale, premature and future requests cannot exempt worker lifetime."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    record = _intent_record(parent, pid, now)
    record["requested_monotonic_s"] = now + offset
    with pytest.raises(ValueError):
        _validate(record, parent, now)


@pytest.mark.fast
@pytest.mark.parametrize("field", ["heartbeat_monotonic_s", "job_started_monotonic_s"])
@pytest.mark.parametrize(
    "bad", [True, float("nan"), float("inf"), -1, "future", None, "invalid"]
)
def test_bad_worker_clocks_remain_conservative_during_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_identity: tuple[int, int],
    field: str,
    bad: object,
) -> None:
    """No authorization context rehabilitates invalid worker telemetry."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    monkeypatch.setattr(status.time, "monotonic", lambda: now)
    intent = _validate(_intent_record(parent, pid, now), parent, now)
    record = _worker_record(parent, pid, now)
    record[field] = now + 1 if bad == "future" else bad
    reading = _reading(tmp_path, pid, record)
    assert reading.active_age(intent, now) == 7199.5
    assert reading.shutdown_failure(intent, now) == "invalid_persistent_worker_status"


@pytest.mark.fast
@pytest.mark.parametrize(
    "damage", ["missing", "oversized", "padded", "stale_stop", "reversed"]
)
def test_shutdown_intent_never_repairs_bad_or_stale_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_identity: tuple[int, int],
    damage: str,
) -> None:
    """Valid supervisor evidence cannot authorize absent or pre-request telemetry."""
    parent, pid = owned_identity
    now = time.monotonic() + 7200
    monkeypatch.setattr(status.time, "monotonic", lambda: now)
    intent = _validate(_intent_record(parent, pid, now), parent, now)
    record = _worker_record(parent, pid, now)
    if damage == "stale_stop":
        record["heartbeat_monotonic_s"] = intent.requested - 0.1
    elif damage == "reversed":
        record["job_started_monotonic_s"] = now
    elif damage == "oversized":
        record["extra"] = "x" * status.MAX_STATUS_BYTES
    reading = (
        status.read_job_status(tmp_path, pid, 7199.5)
        if damage == "missing"
        else _reading(tmp_path, pid, record)
    )
    if damage == "padded":
        path = tmp_path / "pipeline/worker_processes/training_worker.json"
        path.write_text(path.read_text() + " " * status.MAX_STATUS_BYTES)
        reading = status.read_job_status(tmp_path, pid, 7199.5)
    assert reading.active_age(intent, now) == 7199.5
    assert reading.shutdown_failure(intent, now) == (
        "stale_persistent_worker_stop"
        if damage == "stale_stop"
        else "invalid_persistent_worker_status"
    )


@pytest.mark.fast
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1])
def test_invalid_monitor_clocks_are_not_deadline_evidence(value: float) -> None:
    """Native elapsed is finite numeric evidence, never a boolean or cleanup inference."""
    sample = dict(
        available_ram_mb=30000.0,
        free_disk_mb=50000.0,
        group_rss_mb=1.0,
        workspace_mb=1.0,
        scratch_mb=1.0,
        longest_worker_seconds=1.0,
    )
    with pytest.raises(ValueError):
        operator.resource_limit_reason(MorpionContinualLearningPolicy(), sample, value)


@pytest.mark.fast
@pytest.mark.parametrize(
    "ignore_term,early,killed_worker,late_request",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (False, True, False, True),
    ],
)
def test_actual_supervisor_handoff_and_worker_descriptor_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ignore_term: bool,
    early: bool,
    killed_worker: bool,
    late_request: bool,
) -> None:
    """Run the real shell/monitor with inert workers, never scientific stages."""
    stub = tmp_path / "python_stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        """
import json,os,signal,sys,time
from pathlib import Path
if "-c" in sys.argv:
    code=sys.argv[sys.argv.index("-c")+1]
    if "control.recv" in code:
        if os.environ["TEST_LATE"]=="1":
            work=Path(os.environ["MORPION_WORK_DIR"])
            (work/"helper_waiting").write_text("ready")
            while not (work/"native_decision").exists():time.sleep(.01)
        os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
    if "latest_tree_snapshot_path" in code:print("fixture-source");sys.exit(0)
    sys.exit(1)
assert "--persistent-worker" in sys.argv
assert "MORPION_SUPERVISOR_SHUTDOWN_FD" not in os.environ
for fd in Path("/proc/self/fd").iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert target!="socket:["+os.environ["TEST_CONTROL_INODE"]+"]"
from chipiron.environments.morpion.bootstrap.persistent_worker_state import WorkerJobState
stage=sys.argv[sys.argv.index("--pipeline-stage")+1]
state=WorkerJobState(Path(os.environ["MORPION_WORK_DIR"]),stage)
def stop(*_args):
    if os.environ["TEST_EARLY"]=="1":
        decisions=list((Path(os.environ["MORPION_WORK_DIR"])/"resource_monitor").glob("decision_*.json"))
        assert len(decisions)==1
        assert json.loads(decisions[0].read_text())["reason"]=="supervisor_deadline"
    state.update("stopped",exit_code=143);state.close();sys.exit(143)
signal.signal(signal.SIGTERM,signal.SIG_IGN if os.environ["TEST_IGNORE_TERM"]=="1" else stop)
state.update("idle")
while True:state.update("idle");time.sleep(.05)
"""
    )
    stub.chmod(0o755)
    monkeypatch.setenv("MORPION_PERSISTENT_WORKERS", "1")
    monkeypatch.setenv("MORPION_SEED_MODELS_FROM_WORK_DIR", "")
    (tmp_path / "run_state.json").write_text(
        '{"latest_tree_snapshot_path":"fixture-source"}'
    )
    original_popen = subprocess.Popen

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        fd = int(kwargs["env"]["MORPION_SUPERVISOR_SHUTDOWN_FD"])
        kwargs["env"] = dict(
            kwargs["env"],
            PYTHON_BIN=str(stub),
            TEST_CONTROL_INODE=str(os.fstat(fd).st_ino),
            TEST_IGNORE_TERM="1" if ignore_term else "0",
            TEST_EARLY="1" if early and not late_request else "0",
            TEST_LATE="1" if late_request else "0",
        )
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(operator.subprocess, "Popen", launch)
    monkeypatch.setattr(operator, "available_ram_mb", lambda: 20000.0)
    monkeypatch.setattr(
        operator,
        "load_bootstrap_run_state",
        lambda _p: SimpleNamespace(metadata={}, tree_size_at_last_save=1),
    )
    monkeypatch.setattr(operator, "publication_records", lambda _meta: [])
    monitor = tmp_path / "resource_monitor"
    monitor.mkdir()
    real_sleep = time.sleep
    if early:
        (monitor / "usage.json").write_text('{"elapsed_seconds":7196.1}')
        monkeypatch.setattr(operator.time, "sleep", lambda _s: real_sleep(0.01))
        original_send = operator.socket.socket.sendall

        def send_ack(sock: Any, payload: bytes) -> None:
            """Exercise a real stream ACK fragmented across all three bytes."""
            for byte in payload:
                original_send(sock, bytes([byte]))
                real_sleep(0.02)

        monkeypatch.setattr(operator.socket.socket, "sendall", send_ack)
    if late_request:
        original_atomic = operator.atomic_json
        original_children = psutil.Process.children

        def publish_native_decision(path: Path, value: dict[str, Any]) -> None:
            """Release the actual helper only after native sampling has ended."""
            original_atomic(path, value)
            if path.name.startswith("decision_"):
                assert (tmp_path / "helper_waiting").exists()
                (tmp_path / "native_decision").touch()

        def slow_cleanup_children(process: Any, recursive: bool = False) -> Any:
            """Expose close-before-TERM races without slowing the sampler."""
            if not recursive:
                real_sleep(0.2)
            return original_children(process, recursive=recursive)

        monkeypatch.setattr(operator, "atomic_json", publish_native_decision)
        monkeypatch.setattr(psutil.Process, "children", slow_cleanup_children)
    if killed_worker:
        original_atomic = operator.atomic_json

        def persist_then_kill(path: Path, value: dict[str, Any]) -> None:
            """Reproduce death after intent liveness checks but before ACK."""
            original_atomic(path, value)
            if path.name.startswith("decision_"):
                record = json.loads(
                    (
                        tmp_path / "pipeline/worker_processes/training_worker.json"
                    ).read_text()
                )
                os.kill(record["identity"]["pid"], signal.SIGKILL)

        monkeypatch.setattr(operator, "atomic_json", persist_then_kill)
    script = (
        Path(__file__).resolve().parents[2] / "scripts/launch_morpion_gnome_cluster.sh"
    )
    with ExitStack() as resources:

        def run() -> None:
            """Exercise installed native monitoring and the real shell cleanup."""
            operator._run_monitored(  # pylint: disable=protected-access
                MorpionBootstrapPaths.from_work_dir(tmp_path),
                script,
                MorpionContinualLearningPolicy(
                    max_elapsed_seconds=7200 if early else 4,
                    worker_max_seconds=1800,
                    shutdown_grace_seconds=3 if early else 2,
                ),
                monitor,
                psutil,
                resources,
            )

        if ignore_term or killed_worker:
            with pytest.raises(ValueError, match="Persistent worker group failed"):
                run()
        else:
            run()
    receipt = next(
        value
        for path in monitor.glob("session_*.json")
        if "policy" in (value := json.loads(path.read_text()))
    )
    decision = receipt["decision"]
    assert decision["reason"] in {"maximum_elapsed_time", "supervisor_deadline"}
    assert receipt["shutdown_confirmed"]
    assert receipt["shutdown_grace_expired"] == ignore_term
    assert receipt["reason"] == (
        "supervisor_shutdown_grace_expired"
        if ignore_term and decision["reason"] == "supervisor_deadline"
        else "supervisor_failed_during_shutdown"
        if killed_worker
        else decision["reason"]
    )
    if ignore_term:
        assert receipt["supervisor_exit_code"] in {124, -signal.SIGKILL}
    if killed_worker:
        assert receipt["supervisor_exit_code"] == 1
    if early and not late_request:
        assert decision["reason"] == "supervisor_deadline"
        assert decision["supervisor_shutdown"] is not None
        assert decision["budget_elapsed_seconds"] < 7200
    if late_request:
        assert decision["reason"] == "maximum_elapsed_time"
        assert decision["supervisor_shutdown"] is None
    assert (decision["budget_elapsed_seconds"] >= (7200 if early else 4)) == (
        decision["reason"] == "maximum_elapsed_time"
    )
    for stage in ("growth", "dataset_worker", "training_worker", "reevaluation"):
        record = json.loads(
            (tmp_path / "pipeline/worker_processes" / f"{stage}.json").read_text()
        )
        if not ignore_term and not (killed_worker and stage == "training_worker"):
            assert record["phase"] == "stopped" and record["exit_code"] == 143
        assert not psutil.pid_exists(record["identity"]["pid"])
