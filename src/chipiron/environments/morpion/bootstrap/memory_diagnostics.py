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
import warnings
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatchcase
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
MAX_REFERRER_REPR_CHARS = 300
DEFAULT_REFERRER_TYPE_PATTERNS = (
    "anemone.checkpoints.payloads.AlgorithmNodeCheckpointPayload",
    "anemone.checkpoints.payloads.LinkedChildCheckpointPayload",
    "anemone.checkpoints.payloads.NodeEvaluationCheckpointPayload",
    "anemone.checkpoints.payloads.BackupRuntimeCheckpointPayload",
    "anemone.checkpoints.payloads.SerializedValuePayload",
)


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticsConfig:
    """Configuration for optional bootstrap memory diagnostics."""

    enabled: bool = False
    gc_growth: bool = False
    tracemalloc: bool = False
    torch_tensors: bool = False
    referrers: bool = False
    referrer_type_patterns: tuple[str, ...] = ()
    referrer_max_objects_per_type: int = 2
    referrer_max_depth: int = 2
    referrer_top_n: int = 20
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
            if self._config.torch_tensors:
                self._log_torch_tensors(tag)
            if self._config.referrers:
                self._log_referrers(tag)
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
            delta = (
                None if previous_counts is None else count - previous_counts[type_name]
            )
            LOGGER.info(
                "[memory_gc_type] tag=%s type=%s count=%s delta=%s",
                tag,
                type_name,
                count,
                delta,
            )

    def _log_torch_tensors(self, tag: str) -> None:
        try:
            import torch
        except ImportError:
            LOGGER.warning("[memory_torch] tag=%s torch_unavailable=true", tag)
            return

        gc.collect()
        tensor_count = 0
        total_bytes = 0
        storage_ptrs: set[tuple[str, int]] = set()
        storage_bytes = 0
        by_kind: Counter[str] = Counter()
        for obj in gc.get_objects():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    is_tensor = torch.is_tensor(obj)
            except Exception:
                continue
            if not is_tensor:
                continue
            tensor = obj
            tensor_count += 1
            try:
                tensor_bytes = tensor.numel() * tensor.element_size()
                total_bytes += tensor_bytes
                by_kind[
                    f"device={tensor.device} dtype={tensor.dtype} "
                    f"requires_grad={tensor.requires_grad}"
                ] += tensor_bytes
                storage = tensor.untyped_storage()
                storage_key = (str(tensor.device), int(storage.data_ptr()))
                if storage_key not in storage_ptrs:
                    storage_ptrs.add(storage_key)
                    storage_bytes += storage.nbytes()
            except Exception:
                by_kind["uninspectable"] += 1
        LOGGER.info(
            "[memory_torch] tag=%s tensors=%s tensor_bytes_mb=%.3f "
            "unique_storages=%s storage_bytes_mb=%.3f",
            tag,
            tensor_count,
            total_bytes / BYTES_PER_MIB,
            len(storage_ptrs),
            storage_bytes / BYTES_PER_MIB,
        )
        for kind, byte_count in by_kind.most_common(max(self._config.top_n, 0)):
            LOGGER.info(
                "[memory_torch_kind] tag=%s kind=%s bytes_mb=%.3f",
                tag,
                kind,
                byte_count / BYTES_PER_MIB,
            )

    def _log_referrers(self, tag: str) -> None:
        try:
            self._log_referrers_impl(tag)
        except Exception:
            LOGGER.exception("[memory_referrers] tag=%s diagnostics_failed=true", tag)

    def _log_referrers_impl(self, tag: str) -> None:
        patterns = self._config.referrer_type_patterns or DEFAULT_REFERRER_TYPE_PATTERNS
        max_objects_per_type = max(self._config.referrer_max_objects_per_type, 0)
        max_depth = max(self._config.referrer_max_depth, 0)
        top_n = max(self._config.referrer_top_n, 0)
        if max_objects_per_type == 0 or top_n == 0:
            LOGGER.info(
                "[memory_referrers] tag=%s skipped=true reason=empty_limits",
                tag,
            )
            return

        gc.collect()
        objects_by_type: dict[str, list[object]] = {}
        type_counts: Counter[str] = Counter()
        objects = gc.get_objects()
        try:
            diagnostic_container_ids = {
                id(objects),
                id(objects_by_type),
                id(type_counts),
                id(patterns),
            }
            for obj in objects:
                type_name = _object_type_key(obj)
                if not _matches_type_patterns(type_name, patterns):
                    continue
                type_counts[type_name] += 1
                examples = objects_by_type.setdefault(type_name, [])
                if len(examples) < max_objects_per_type:
                    examples.append(obj)

            LOGGER.info(
                "[memory_referrers] tag=%s matched_types=%s matched_objects=%s "
                "patterns=%s max_objects_per_type=%s max_depth=%s",
                tag,
                len(type_counts),
                sum(type_counts.values()),
                patterns,
                max_objects_per_type,
                max_depth,
            )
            for type_name, count in type_counts.most_common(top_n):
                examples = objects_by_type.get(type_name, [])
                diagnostic_container_ids.add(id(examples))
                LOGGER.info(
                    "[memory_referrer_type] tag=%s type=%s count=%s inspected=%s",
                    tag,
                    type_name,
                    count,
                    len(examples),
                )
                for index, obj in enumerate(examples, start=1):
                    object_path = f"{type_name}[{index}]"
                    LOGGER.info(
                        "[memory_referrer_object] tag=%s path=%s object_id=%s repr=%s",
                        tag,
                        object_path,
                        hex(id(obj)),
                        _safe_repr(obj),
                    )
                    self._log_referrer_tree(
                        tag=tag,
                        target=obj,
                        path=object_path,
                        depth=1,
                        max_depth=max_depth,
                        top_n=top_n,
                        seen_ids={id(obj)},
                        diagnostic_container_ids=diagnostic_container_ids,
                    )
        finally:
            for examples in objects_by_type.values():
                examples.clear()
            objects_by_type.clear()
            type_counts.clear()
            objects.clear()

    def _log_referrer_tree(
        self,
        *,
        tag: str,
        target: object,
        path: str,
        depth: int,
        max_depth: int,
        top_n: int,
        seen_ids: set[int],
        diagnostic_container_ids: set[int],
    ) -> None:
        if depth > max_depth:
            return
        referrers = gc.get_referrers(target)
        local_container_ids = {
            *diagnostic_container_ids,
            id(referrers),
            id(seen_ids),
        }
        logged_count = 0
        try:
            for referrer in referrers:
                if _should_skip_referrer(referrer, local_container_ids):
                    continue
                referrer_id = id(referrer)
                if referrer_id in seen_ids:
                    continue
                logged_count += 1
                referrer_path = f"{path}.ref{logged_count}"
                LOGGER.info(
                    "[memory_referrer] tag=%s path=%s depth=%s ref_type=%s "
                    "ref_id=%s repr=%s",
                    tag,
                    referrer_path,
                    depth,
                    _object_type_key(referrer),
                    hex(referrer_id),
                    _safe_repr(referrer),
                )
                seen_ids.add(referrer_id)
                try:
                    self._log_referrer_tree(
                        tag=tag,
                        target=referrer,
                        path=referrer_path,
                        depth=depth + 1,
                        max_depth=max_depth,
                        top_n=top_n,
                        seen_ids=seen_ids,
                        diagnostic_container_ids=local_container_ids,
                    )
                finally:
                    seen_ids.discard(referrer_id)
                if logged_count >= top_n:
                    break
        finally:
            referrers.clear()

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


def _matches_type_patterns(type_name: str, patterns: tuple[str, ...]) -> bool:
    return any(
        pattern == type_name or pattern in type_name or fnmatchcase(type_name, pattern)
        for pattern in patterns
    )


def _should_skip_referrer(
    referrer: object,
    diagnostic_container_ids: set[int],
) -> bool:
    if id(referrer) in diagnostic_container_ids:
        return True
    return isinstance(referrer, FrameType)


def _safe_repr(obj: object, max_chars: int = MAX_REFERRER_REPR_CHARS) -> str:
    try:
        rendered = repr(obj)
    except Exception as exc:
        rendered = f"<repr_failed {type(exc).__module__}.{type(exc).__qualname__}>"
    rendered = " ".join(rendered.splitlines())
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[: max_chars - 3]}..."


__all__ = [
    "DEFAULT_REFERRER_TYPE_PATTERNS",
    "MemoryDiagnostics",
    "MemoryDiagnosticsConfig",
]
