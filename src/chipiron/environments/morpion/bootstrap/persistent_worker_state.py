"""Bounded persistent-process status for the existing resource monitor.

Status never authorizes claims or takeover. Missing, stale or mismatched evidence
falls back to process age; only a live matching idle record excludes idle time.
"""

import fcntl
import math
import os
import time
from pathlib import Path
from typing import Any

from .derived.provenance import atomic_json
from .generation_publication_io import require
from .learning_worker import monitored_identity

STAGES = frozenset({"growth", "dataset_worker", "training_worker", "reevaluation"})
MAX_STATUS_BYTES = 16 * 1024


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


def active_job_age(work_dir: Path, pid: int, process_age: float) -> float:
    """Use matching active-job age or fresh idle evidence; otherwise fail conservatively."""
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
                data = json.loads(stream.read(MAX_STATUS_BYTES + 1))
            identity = data["identity"]
            if identity["pid"] != pid or identity["start_ticks"] != fields[19]:
                continue
            require(
                data["schema_version"] == 1
                and data["worker"] == worker
                and identity["session"] == int(fields[3])
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
                all(
                    isinstance(x, (float, int))
                    and not isinstance(x, bool)
                    and math.isfinite(x)
                    and 0 <= x <= now
                    for x in (heartbeat, started)
                ),
                "Invalid worker status clock",
            )
            if data["phase"] == "idle" and now - heartbeat <= 5:
                return 0.0
            if data["phase"] in {"starting", "running", "cleanup"}:
                return now - float(started)
    except OSError, ValueError, KeyError, TypeError, IndexError:
        return process_age
    return process_age
