"""Bounded persistent-process status for the existing resource monitor.

Status never authorizes claims or takeover. Missing, stale or mismatched evidence
falls back to process age; only a live matching idle record excludes idle time.
"""

import fcntl
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .derived.provenance import atomic_json
from .generation_publication_io import require
from .learning_worker import monitored_identity

STAGES = frozenset({"growth", "dataset_worker", "training_worker", "reevaluation"})
MAX_STATUS_BYTES = 16 * 1024


def _clock(value: object, now: float) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= now
    except OverflowError:
        return False


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SupervisorShutdown:
    """Validated operational intent received on the supervisor-only control socket."""

    record: dict[str, Any]

    @property
    def requested(self) -> float:
        """Return the pre-signal monotonic decision clock, not cleanup elapsed."""
        return float(self.record["requested_monotonic_s"])

    @property
    def deadline(self) -> float:
        """Bound the requested shutdown, including waiting for acknowledgement."""
        return self.requested + float(self.record["grace_seconds"])


def validate_shutdown_intent(
    record: dict[str, Any],
    *,
    monitor_session: str,
    parent_pid: int,
    limit_seconds: int,
    grace_seconds: int,
    not_before: float,
    now: float,
) -> SupervisorShutdown:
    """Bind a control-socket message to this live supervisor and declared budget.

    File contents or worker exit codes are never shutdown authority. The caller
    owns the socket; workers must not inherit its supervisor endpoint. The shell
    waits for acknowledgement before signalling, so its kernel identity is live.
    """
    pid = record["supervisor_pid"]
    require(_integer(pid) and pid > 0, "Invalid shutdown supervisor PID")
    fields = (
        Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
    )
    own = Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
    requested = record["requested_monotonic_s"]
    require(
        _integer(record["schema_version"])
        and record["schema_version"] == 1
        and record["monitor_session"] == monitor_session
        and _integer(record["monitor_pid"])
        and record["monitor_pid"] == os.getpid()
        and record["monitor_start_ticks"] == own[19]
        and record["boot"]
        == Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        and record["pid_namespace"] == os.readlink("/proc/self/ns/pid")
        and fields[0] not in {"Z", "X"}
        and int(fields[1]) == parent_pid
        and fields[19] == record["supervisor_start_ticks"]
        and _integer(record["supervisor_session"])
        and int(fields[3]) == record["supervisor_session"] == parent_pid
        and record["reason"] == "supervisor_deadline"
        and _integer(record["limit_seconds"])
        and record["limit_seconds"] == limit_seconds
        and _integer(record["grace_seconds"])
        and record["grace_seconds"] == grace_seconds
        and _clock(requested, now)
        and not_before <= requested
        and now <= requested + grace_seconds,
        "Invalid, stale or mismatched supervisor shutdown intent",
    )
    return SupervisorShutdown(record)


@dataclass(frozen=True)
class WorkerJobStatus:
    """A bounded identity-checked observation; invalid telemetry retains process age."""

    age: float
    record: dict[str, Any] | None = None

    def shutdown_failure(
        self, intent: SupervisorShutdown | None, now: float
    ) -> str | None:
        """Never infer intentional shutdown from a stopped phase or exit143 alone."""
        if self.record is None:
            return "invalid_persistent_worker_status" if intent is not None else None
        if self.record["phase"] != "stopped":
            return None
        code = self.record.get("exit_code")
        if not _integer(code) or code not in {0, 143}:
            return "persistent_worker_failed"
        if intent is None:
            return "unexpected_persistent_worker_stop"
        if now > intent.deadline:
            return "supervisor_shutdown_grace_expired"
        if self.record["heartbeat_monotonic_s"] < intent.requested:
            return "stale_persistent_worker_stop"
        return None

    def active_age(self, intent: SupervisorShutdown | None, now: float) -> float:
        """Freeze an authorized stopped job at its terminal heartbeat, not process birth."""
        if self.record is not None and self.record["phase"] in {
            "starting",
            "running",
            "cleanup",
        }:
            return now - float(self.record["job_started_monotonic_s"])
        if (
            self.record is not None
            and self.record["phase"] == "stopped"
            and intent is not None
            and self.shutdown_failure(intent, now) is None
        ):
            return float(self.record["heartbeat_monotonic_s"]) - float(
                self.record["job_started_monotonic_s"]
            )
        return self.age


def _status_path(work_dir: Path, worker: str) -> Path:
    require(worker in STAGES, "Unknown persistent worker stage")
    return work_dir / "pipeline" / "worker_processes" / f"{worker}.json"


class WorkerJobState:
    """One atomic current record per worker; scientific authority remains elsewhere."""

    def __init__(self, work_dir: Path, worker: str) -> None:
        """Acquire exclusive process registration and publish initial identity."""
        self.path = _status_path(work_dir, worker)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.identity = monitored_identity()
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "worker": worker,
            "identity": self.identity,
            "process_started_monotonic_s": time.monotonic(),
            "job_id": 0,
            "phase": "starting",
            "job_started_monotonic_s": time.monotonic(),
            "heartbeat_scope": "job boundaries and idle polling; active jobs use elapsed bound",
        }
        self.update("starting")

    def close(self) -> None:
        """Release only operational process registration, never scientific ownership."""
        self.lock.close()

    def update(self, phase: str, **details: object) -> None:
        """Publish supervision state; failures are fatal, never silently idle."""
        require(
            phase in {"starting", "running", "cleanup", "idle", "stopped"},
            "Invalid worker phase",
        )
        require(
            monitored_identity() == self.identity, "Persistent monitor identity changed"
        )
        self.record.update(details)
        self.record.update(
            phase=phase,
            heartbeat_monotonic_s=time.monotonic(),
            updated_unix_s=time.time(),
        )
        atomic_json(self.path, self.record)


def read_job_status(work_dir: Path, pid: int, process_age: float) -> WorkerJobStatus:
    """Read real identity and clocks once; retain evidence for the pre-cleanup decision."""
    import json

    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()
        )
        for worker in STAGES:
            path = _status_path(work_dir, worker)
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                payload = stream.read(MAX_STATUS_BYTES + 1)
            require(len(payload) <= MAX_STATUS_BYTES, "Oversized worker status")
            data = json.loads(payload)
            identity = data["identity"]
            if identity["pid"] != pid or identity["start_ticks"] != fields[19]:
                continue
            require(
                _integer(data["schema_version"])
                and data["schema_version"] == 1
                and data["worker"] == worker
                and _integer(identity["pid"])
                and _integer(identity["session"])
                and identity["session"] == int(fields[3])
                and _integer(identity["monitor_pid"])
                and identity["monitor_pid"] == os.getpid()
                and identity["monitor_start_ticks"]
                == Path("/proc/self/stat")
                .read_text(encoding="utf-8")
                .rsplit(") ", 1)[1]
                .split()[19]
                and identity["pid_namespace"] == os.readlink("/proc/self/ns/pid")
                and identity["boot"]
                == Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="utf-8")
                .strip(),
                "Persistent worker status identity differs",
            )
            heartbeat = data["heartbeat_monotonic_s"]
            started = data["job_started_monotonic_s"]
            # A concurrently published record can be newer than a timestamp
            # captured before its read. Truly future clocks still fail closed.
            now = time.monotonic()
            require(
                all(_clock(x, now) for x in (heartbeat, started))
                and started <= heartbeat,
                "Invalid worker status clock",
            )
            if data["phase"] == "idle" and now - heartbeat <= 5:
                return WorkerJobStatus(0.0, data)
            if data["phase"] in {"starting", "running", "cleanup"}:
                return WorkerJobStatus(now - float(started), data)
            if data["phase"] == "stopped":
                return WorkerJobStatus(process_age, data)
    except OSError, ValueError, KeyError, TypeError, IndexError, RecursionError:
        return WorkerJobStatus(process_age)
    return WorkerJobStatus(process_age)


def active_job_age(work_dir: Path, pid: int, process_age: float) -> float:
    """Preserve conservative legacy accounting without supervisor-only evidence."""
    return read_job_status(work_dir, pid, process_age).age
