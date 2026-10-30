"""Passive bounded boundary records using the existing performance clock schema.

No locks, scheduling, admission or scientific reads. Wait intervals are sampled
between worker decisions, including polling/restart overhead, not CPU blocking.
"""

import hashlib
import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from .performance import StageMeasurement, process_identity

LOGGER = logging.getLogger(__name__)
RUNTIME_SLOTS = 256
MAX_RUNTIME_BYTES = 16 * 1024


def _write(work_dir: Path, name: str, record: dict[str, Any]) -> None:
    """Bound per-record bytes and replace one slot; observational failures stay local."""
    temporary = None
    try:
        directory = work_dir / "pipeline" / "performance" / "runtime"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        raw = json.dumps(
            {"schema_version": 1, "retention_slots": RUNTIME_SLOTS, **record},
            allow_nan=False,
        )
        if len(raw.encode()) > MAX_RUNTIME_BYTES:
            return
        temporary.write_text(raw, encoding="utf-8")
        temporary.replace(path)
    # This observational boundary must contain every ordinary metrics failure.
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.debug("Runtime metrics unavailable", exc_info=True)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _slot(identity: str) -> int:
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % RUNTIME_SLOTS


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_RUNTIME_BYTES + 1)
        value = json.loads(raw) if len(raw) <= MAX_RUNTIME_BYTES else {}
        return value if isinstance(value, dict) else {}
    except OSError, ValueError:
        return {}


def _boot_id() -> str | None:
    try:
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError:
        return None


def observe_wait(
    work_dir: Path, worker: str, reason: str | None, generation: int | None
) -> None:
    """Coalesce repeated decisions; keep open waits incomplete, never infer their end."""
    try:
        # The shell's creation ticks distinguish sequential invocations from PID reuse.
        parent = os.getppid()
        ticks = (
            Path(f"/proc/{parent}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()[19]
        )
        stream_id = f"{worker}:{parent}:{ticks}"
        cursor_name = f"wait-current-{worker}-{_slot(stream_id)}"
        cursor = (
            work_dir / "pipeline" / "performance" / "runtime" / f"{cursor_name}.json"
        )
        previous = _read(cursor)
        now, monotonic, boot = time.time(), time.monotonic(), _boot_id()
        same_stream = (
            previous.get("stream_id") == stream_id
            and boot is not None
            and previous.get("boot_id") == boot
        )
        if same_stream and previous.get("reason"):
            previous.update(
                last_observed_unix_s=now,
                observed_elapsed_s=monotonic - previous["started_monotonic_s"],
            )
            if reason != previous["reason"] or generation != previous.get("generation"):
                previous.update(
                    finished_unix_s=now,
                    elapsed_s=previous["observed_elapsed_s"],
                    evidence_status="completed",
                    outcome="decision_changed",
                )
                _write(
                    work_dir,
                    f"wait-{worker}-{_slot(str(previous['observation_id']))}",
                    previous,
                )
            else:
                _write(work_dir, cursor_name, previous)
                return
        if reason is None:
            _write(
                work_dir,
                cursor_name,
                {
                    "stream_id": stream_id,
                    "worker": worker,
                    "reason": None,
                    "boot_id": boot,
                },
            )
            return
        _write(
            work_dir,
            cursor_name,
            {
                "stage": "wait",
                "worker": worker,
                "generation": generation,
                "reason": reason,
                "started_unix_s": now,
                "started_monotonic_s": monotonic,
                "last_observed_unix_s": now,
                "observed_elapsed_s": 0.0,
                "elapsed_s": None,
                "finished_unix_s": None,
                "evidence_status": "incomplete",
                "stream_id": stream_id,
                "boot_id": boot,
                "observation_id": f"{stream_id}:{time.time_ns()}",
                "timing_scope": "sampled_worker_decisions_including_poll_and_restart_overhead",
                "pid": os.getpid(),
                "process_identity": process_identity(),
            },
        )
    # Reporting must never change the scheduling decision it observes.
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.debug("Wait metrics unavailable", exc_info=True)


def observe_ownership(
    work_dir: Path,
    *,
    started_unix_s: float,
    started: float,
    acquired: float | None,
    finished: float,
    outcome: str,
) -> None:
    """Record actual existing monotonic ownership boundaries after release/timeout."""
    try:
        waited = (acquired if acquired is not None else finished) - started
        record = {
            "stage": "ownership",
            "generation": None,
            "pid": os.getpid(),
            "process_identity": process_identity(),
            "reason": "ownership_contention"
            if waited >= 0.05
            else "ownership_acquisition",
            "started_unix_s": started_unix_s,
            "finished_unix_s": time.time(),
            "elapsed_s": finished - started,
            "outcome": outcome,
            "evidence_status": "completed" if acquired is not None else "failed",
            "wait_s": waited,
            "held_s": finished - acquired if acquired is not None else None,
            "timing_scope": "ownership_wait_and_hold; wait_s and held_s are disjoint",
        }
        _write(
            work_dir, f"ownership-{_slot(f'{process_identity()}:{started}')}", record
        )
    # Ownership has already released; reporting must not replace its outcome.
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.debug("Ownership metrics unavailable", exc_info=True)


class WorkerObservation:
    """Retain an incomplete invocation and replace it on exit, without changing exit behavior."""

    def __init__(self, work_dir: Path, worker: str) -> None:
        """Start one best-effort dispatch interval without acquiring scientific ownership."""
        self.work_dir, self.worker = work_dir, worker
        self.measurement: StageMeasurement | None = None
        self.name = f"worker-{worker}-{_slot(process_identity())}"
        try:
            self.measurement = StageMeasurement()
            self._startup()
            _write(
                work_dir,
                self.name,
                {
                    "stage": "worker_lifetime",
                    "generation": None,
                    "worker": worker,
                    "pid": os.getpid(),
                    "process_identity": process_identity(),
                    "started_unix_s": self.measurement.started_unix_s,
                    "evidence_status": "incomplete",
                    "timing_scope": "launcher_dispatch_to_exit; startup/imports excluded",
                },
            )
        # A metrics failure must not prevent dispatch of the scientific worker.
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.debug("Worker metrics unavailable", exc_info=True)

    def _startup(self) -> None:
        """Linux process age to dispatch; Unix placement is explicitly boot-clock anchored."""
        try:
            ticks = float(
                Path("/proc/self/stat")
                .read_text(encoding="utf-8")
                .rsplit(") ", 1)[1]
                .split()[19]
            )
            tick_rate = os.sysconf("SC_CLK_TCK")
            boot_now = time.clock_gettime(time.CLOCK_BOOTTIME)
            end = time.time()
            elapsed = boot_now - ticks / tick_rate
            if elapsed < 0:
                return
            _write(
                self.work_dir,
                self.name.replace("worker-", "startup-", 1),
                {
                    "stage": "startup",
                    "generation": None,
                    "worker": self.worker,
                    "pid": os.getpid(),
                    "process_identity": process_identity(),
                    "started_unix_s": end - elapsed,
                    "finished_unix_s": end,
                    "elapsed_s": elapsed,
                    "cpu_s": time.process_time(),
                    "cpu_scope": "current_process_since_birth",
                    "evidence_status": "completed",
                    "timing_scope": "process_birth_to_dispatch; includes interpreter/imports/configuration",
                    "clock_source": "Linux start ticks / CLOCK_BOOTTIME; Unix placement anchored at dispatch",
                    "clock_resolution_s": 1 / tick_rate,
                },
            )
        # Startup clocks are optional diagnostics, never a launch prerequisite.
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.debug("Startup clock unavailable", exc_info=True)

    def finish(self, exit_code: int) -> None:
        """Replace the retained start with the observed outcome; preserve the caller exit."""
        try:
            if self.measurement is not None:
                _write(
                    self.work_dir,
                    self.name,
                    {
                        **self.measurement.finish(),
                        "stage": "worker_lifetime",
                        "generation": None,
                        "worker": self.worker,
                        "outcome": "completed" if exit_code == 0 else "failed",
                        "evidence_status": "completed" if exit_code == 0 else "failed",
                        "timing_scope": "launcher_dispatch_to_exit; startup/imports excluded",
                    },
                )
        # Preserve the worker's original exit code or scientific exception.
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.debug("Worker metrics unavailable", exc_info=True)
