"""Boundary-only resource observations; never read or alter scientific state.

RSS peaks are process-lifetime high water marks, not stage-local maxima. CUDA
peaks are reset only for individual evaluators/inference windows. Their wall
clock includes synchronization at the two boundaries, never inside a batch.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from .pipeline_memory import available_ram_mb, current_rss_mb

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def process_peak_rss_mb() -> float | None:
    """Return the process lifetime RSS high water mark in MiB without polling."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (AttributeError, OSError, ValueError):
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
            except (RuntimeError, ValueError, AssertionError):
                LOGGER.warning("CUDA measurement unavailable", exc_info=True)
                self.cuda = None
        self.started_unix_s = time.time()
        self.started = time.perf_counter()
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
            except (RuntimeError, ValueError, AssertionError):
                LOGGER.warning("CUDA measurement unavailable", exc_info=True)
        return {
            **details,
            "started_unix_s": self.started_unix_s,
            "finished_unix_s": time.time(),
            "elapsed_s": time.perf_counter() - self.started,
            "pid": os.getpid(),
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
    except (OSError, ValueError, TypeError):
        LOGGER.warning(
            "Could not persist performance observation %s", path, exc_info=True
        )
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
