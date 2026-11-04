"""Boundary-only resource observations; never read or alter scientific state.

RSS peaks are process-lifetime high water marks, not stage-local maxima. CUDA
peaks are reset only for individual evaluators/inference windows. Their wall
clock includes synchronization at the two boundaries, never inside a batch.
"""

import json
import logging
import os
import resource
import sys
import time
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .pipeline_memory import available_ram_mb, current_rss_mb

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

LOGGER = logging.getLogger(__name__)
_IMPORT_TIME_NS = time.time_ns()


@lru_cache(maxsize=16)
def _process_identity(pid: int) -> str:
    """Read process birth once per PID, including after a fork."""
    try:
        ticks = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()[19]
        )
        boot = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError, ValueError, IndexError:
        return f"{pid}:import-{_IMPORT_TIME_NS}"
    return f"{boot}:{pid}:{ticks}"


def process_identity() -> str:
    """Keep forked processes distinct without polling or importing process profilers."""
    return _process_identity(os.getpid())


def process_peak_rss_mb() -> float | None:
    """Return the process lifetime RSS high water mark in MiB without polling."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except AttributeError, OSError, ValueError:
        return None


def reaped_children_cpu_s() -> float | None:
    """CPU of waited-for children only; excludes live children and this process."""
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime
    except AttributeError, OSError, ValueError:
        return None


class StageMeasurement:
    """Capture cheap boundaries, without importing Torch in non-ML workers."""

    def __init__(self, *, cuda_device: str | None = None) -> None:
        """Start a wall-clock interval; optionally reset one CUDA device's peaks."""
        self.cuda: Any = None
        self.device: Any = None
        self.cuda_before: dict[str, object] | None = None
        torch = sys.modules.get("torch")
        if cuda_device and cuda_device != "cpu" and torch is not None:
            try:
                if torch.cuda.is_available():
                    self.cuda = torch.cuda
                    self.device = torch.device(
                        "cuda" if cuda_device == "auto" else cuda_device
                    )
                    self.cuda.synchronize(self.device)
                    self.cuda.reset_peak_memory_stats(self.device)
                    self.cuda_before = self._cuda_snapshot()
            except RuntimeError, ValueError, AssertionError:
                LOGGER.warning("CUDA measurement unavailable", exc_info=True)
                self.cuda = None
        self.started_unix_s = time.time()
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.children_cpu_started = reaped_children_cpu_s()
        self.rss_before_mb = current_rss_mb()
        self.available_before_mb = available_ram_mb()

    def _cuda_snapshot(self) -> dict[str, object]:
        """Read allocator counters; these exclude other processes and driver memory."""
        return {
            "device": str(self.device),
            "device_name": self.cuda.get_device_name(self.device),
            "device_total_bytes": self.cuda.get_device_properties(
                self.device
            ).total_memory,
            "allocated_bytes": self.cuda.memory_allocated(self.device),
            "reserved_bytes": self.cuda.memory_reserved(self.device),
            "max_allocated_bytes": self.cuda.max_memory_allocated(self.device),
            "max_reserved_bytes": self.cuda.max_memory_reserved(self.device),
        }

    def finish(self, **details: object) -> dict[str, object]:
        """Return a JSON-ready observation; no background samplers or RNG calls."""
        cuda_after = None
        if self.cuda is not None:
            try:
                self.cuda.synchronize(self.device)
                cuda_after = self._cuda_snapshot()
            except RuntimeError, ValueError, AssertionError:
                LOGGER.warning("CUDA measurement unavailable", exc_info=True)
        children_cpu = reaped_children_cpu_s()
        return {
            **details,
            "started_unix_s": self.started_unix_s,
            "finished_unix_s": time.time(),
            "elapsed_s": time.perf_counter() - self.started,
            "cpu_s": time.process_time() - self.cpu_started,
            "pid": os.getpid(),
            "process_identity": process_identity(),
            "cpu_scope": "current_process",
            "children_cpu_s": children_cpu - self.children_cpu_started
            if children_cpu is not None and self.children_cpu_started is not None
            else None,
            "children_cpu_scope": "reaped_children_only; live children excluded",
            "rss_peak_scope": "process_lifetime",
            "device_timing_scope": "host_wall_with_device_boundary_synchronization"
            if self.cuda_before is not None and cuda_after is not None
            else "host_wall; device completion unverified",
            "rss_before_mb": self.rss_before_mb,
            "rss_after_mb": current_rss_mb(),
            "process_peak_rss_mb": process_peak_rss_mb(),
            "available_ram_before_mb": self.available_before_mb,
            "available_ram_after_mb": available_ram_mb(),
            "cuda_before": self.cuda_before,
            "cuda_after": cuda_after,
        }


def persist_stage_measurement(
    work_dir: Path,
    generation: int,
    stage: str,
    observation: Mapping[str, object],
) -> None:
    """Atomically publish one small immutable observation, independently per worker.

    Metrics failures are logged without failing or retrying scientific work. Names
    use process/clock identity, never the search or training random streams.
    """
    directory = work_dir / "pipeline" / "performance" / f"generation_{generation:06d}"
    path = directory / f"{stage}-{os.getpid()}-{time.time_ns()}.json"
    temporary = path.with_suffix(".tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            **observation,
            "schema_version": 1,
            "generation": generation,
            "stage": stage,
        }
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    except OSError, ValueError, TypeError:
        LOGGER.warning(
            "Could not persist performance observation %s", path, exc_info=True
        )
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def optional_measurement() -> StageMeasurement | None:
    """Keep ordinary observer failures separate from scientific computation."""
    try:
        return StageMeasurement()  # Host only: never initialize/synchronize a device.
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def optional_finish(measurement: StageMeasurement | None) -> dict[str, object]:
    """Finish an existing observation once; missing evidence is not zero time."""
    try:
        return {} if measurement is None else measurement.finish()
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


class ActivationPhaseTiming:
    """One bounded operational buffer, never a scientific authority or failure log.

    Failed/uncommitted attempts are dropped to preserve rejection byte contracts.
    A completed commit may publish even if the final runtime cleanup raises.
    """

    MAX_ROWS = 64
    MAX_BYTES = 64 * 1024

    def __init__(self, identity: dict[str, object]) -> None:
        """Use already-known scalar identity only; no source/configuration reads."""
        self.identity = identity
        self.source = {
            key: identity.get(key)
            for key in ("source_tree_generation", "source_publication_id")
        }
        self.rows: list[dict[str, object]] = []
        self.dropped = 0
        self.errors = 0
        self.ordinal = 0
        self.invocation: str | None = None
        try:
            self.invocation = f"{process_identity()}:{time.time_ns()}"
        except Exception:  # pylint: disable=broad-exception-caught
            self.errors += 1

    def set_source(self, generation: int, publication_id: str) -> None:
        """Update already-known committed source after a successful drain/save."""
        try:
            self.source = {
                "source_tree_generation": generation,
                "source_publication_id": publication_id,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            self.errors += 1

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        parent: str | None = None,
        outside: bool = False,
        output_generation: int | None = None,
        output_publication_id: str | None = None,
    ) -> Iterator[None]:
        """Bracket a macro call; computation and cancellation are never caught."""
        measurement = None
        row: dict[str, object] = {}
        try:
            measurement = optional_measurement()
            self.ordinal += 1
            row = {
                **self.source,
                "output_generation": output_generation,
                "output_publication_id": output_publication_id,
                "phase": name,
                "ordinal": self.ordinal,
                "parent_phase": parent,
                "outside_activation_total": outside,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            self.errors += 1
        try:
            yield
        finally:
            # Catch only observer work, not the operation inside the context.
            if row:
                try:
                    observation = optional_finish(measurement)
                    if not observation:
                        self.errors += 1
                    row.update(observation)
                    row.update(
                        monotonic_start_s=None
                        if measurement is None
                        else measurement.started,
                        monotonic_end_s=(
                            None
                            if not observation or measurement is None
                            else measurement.started + float(observation["elapsed_s"])  # type: ignore[arg-type]
                        ),
                        outcome="failed"
                        if sys.exc_info()[0] is not None
                        else "completed",
                    )
                    if len(self.rows) < self.MAX_ROWS:
                        self.rows.append(row)
                    else:
                        self.dropped += 1
                except Exception:  # pylint: disable=broad-exception-caught
                    self.errors += 1

    def publish(
        self,
        work_dir: Path,
        generation: int,
        publication_id: str,
        total: dict[str, object],
        *,
        function_failed: bool,
    ) -> None:
        """Emit at most one bounded sidecar; even a failing sink/logger is harmless."""
        try:
            payload: dict[str, object] = {
                **self.identity,
                "schema_version": 1,
                "generation": generation,
                "stage": "activation.phases",
                "phase_schema_version": 1,
                "invocation_id": self.invocation,
                "publication_id": publication_id,
                "scientific_outcome": "committed",
                "function_outcome": "failed" if function_failed else "completed",
                "total": total,
                "phases": list(self.rows),
                "dropped_rows": self.dropped,
                "observer_errors": self.errors,
                "phase_calls": self.ordinal,
            }
            rows = payload["phases"]
            assert isinstance(rows, list)
            while len(json.dumps(payload, allow_nan=False).encode()) > self.MAX_BYTES:
                if not rows:
                    return
                rows.pop()
                payload["dropped_rows"] = int(payload["dropped_rows"]) + 1  # type: ignore[call-overload]
            persist_stage_measurement(
                work_dir, generation, "activation.phases", payload
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # In particular, the existing sink's warning logger can itself fail.
            return
