"""Lightweight process-memory instrumentation for Morpion bootstrap runs.

Example:
    python -m chipiron.environments.morpion.bootstrap.launcher \
      --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
      --max-cycles 1 \
      --max-growth-steps-per-cycle 80 \
      --memory-diagnostics \
      --memory-diagnostics-gc-growth \
      --memory-diagnostics-tracemalloc \
      2>&1 | tee ~/oldata/victor/morpion_runs/big_run_01/memory_debug.log
"""

from __future__ import annotations

import gc
import logging
import os
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticsConfig:
    """Configuration for optional bootstrap memory diagnostics."""

    enabled: bool = False
    gc_growth: bool = False
    tracemalloc: bool = False
    top_n: int = 20


class MemoryDiagnostics:
    """Best-effort memory diagnostics that avoids retaining large objects."""

    def __init__(self, config: MemoryDiagnosticsConfig) -> None:
        """Initialize memory diagnostics from a small immutable config."""
        self._config = config
        self._process: object | None = None
        self._previous_type_counts: Counter[str] | None = None
        self._previous_snapshot: tracemalloc.Snapshot | None = None
        self._started_tracemalloc = False
        self._warned_psutil_unavailable = False
        if not config.enabled:
            return
        self._process = self._build_process()
        if config.tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True

    def __enter__(self) -> MemoryDiagnostics:
        """Enter a context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close diagnostics on context manager exit."""
        self.close()

    def log(self, tag: str) -> None:
        """Log process memory and optional object/allocation growth for ``tag``."""
        if not self._config.enabled:
            return
        try:
            self._log_process_memory(tag)
            if self._config.gc_growth:
                self._log_gc_growth(tag)
            if self._config.tracemalloc:
                self._log_tracemalloc(tag)
        except Exception:
            LOGGER.exception("[memory] diagnostics_failed tag=%s", tag)

    def close(self) -> None:
        """Release diagnostic-only references."""
        self._previous_type_counts = None
        self._previous_snapshot = None
        self._process = None
        if self._started_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started_tracemalloc = False

    def _build_process(self) -> object | None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            LOGGER.warning("[memory] psutil_unavailable rss_vms_logging=false")
            self._warned_psutil_unavailable = True
            return None
        return psutil.Process(os.getpid())

    def _log_process_memory(self, tag: str) -> None:
        if self._process is None:
            if not self._warned_psutil_unavailable:
                self._process = self._build_process()
            return
        memory_info = self._process.memory_info()
        LOGGER.info(
            "[memory] tag=%s rss_mb=%.3f vms_mb=%.3f",
            tag,
            memory_info.rss / BYTES_PER_MIB,
            memory_info.vms / BYTES_PER_MIB,
        )

    def _log_gc_growth(self, tag: str) -> None:
        collected = gc.collect()
        objects = gc.get_objects()
        total_objects = len(objects)
        type_counts = Counter(_object_type_key(obj) for obj in objects)
        previous_counts = self._previous_type_counts
        self._previous_type_counts = type_counts
        LOGGER.info(
            "[memory_gc] tag=%s collected=%s total_objects=%s tracked_types=%s",
            tag,
            collected,
            total_objects,
            len(type_counts),
        )
        for type_name, count in type_counts.most_common(max(self._config.top_n, 0)):
            delta = None if previous_counts is None else count - previous_counts[type_name]
            LOGGER.info(
                "[memory_gc_type] tag=%s type=%s count=%s delta=%s",
                tag,
                type_name,
                count,
                delta,
            )

    def _log_tracemalloc(self, tag: str) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        snapshot = tracemalloc.take_snapshot()
        previous_snapshot = self._previous_snapshot
        self._previous_snapshot = snapshot
        if previous_snapshot is None:
            LOGGER.info("[memory_tracemalloc] tag=%s baseline=true", tag)
            return
        stats = snapshot.compare_to(previous_snapshot, "lineno")
        for index, stat in enumerate(stats[: max(self._config.top_n, 0)], start=1):
            frame = stat.traceback[0]
            LOGGER.info(
                "[memory_tracemalloc] tag=%s rank=%s size_diff_mb=%.6f "
                "count_diff=%s file=%s line=%s",
                tag,
                index,
                stat.size_diff / BYTES_PER_MIB,
                stat.count_diff,
                frame.filename,
                frame.lineno,
            )


def _object_type_key(obj: object) -> str:
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


__all__ = ["MemoryDiagnostics", "MemoryDiagnosticsConfig"]
