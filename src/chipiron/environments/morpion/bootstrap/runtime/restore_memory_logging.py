"""Checkpoint restore memory logging helpers for Morpion runtime."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    current_rss_mb as _pipeline_current_rss_mb,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive_memory import (
    DeepSizeStats,
    deep_size,
)

from .checkpoint_io import _checkpoint_artifact_bytes, _metric_value

if TYPE_CHECKING:
    from pathlib import Path

    from anemone.checkpoints import SearchRuntimeCheckpointPayload

LOGGER = logging.getLogger(
    ".".join(("chipiron.environments.morpion.bootstrap.runtime", "runner"))
)

__all__ = [
    "RestoreMemoryLogger",
    "current_rss_mb",
    "log_morpion_checkpoint_memory_phase",
    "restore_memory_logger_for_checkpoint_path",
]


class _RestoreMemoryArgs(Protocol):
    restore_memory_profile: bool
    restore_memory_profile_recursive: bool
    restore_memory_profile_recursive_max_objects: int | None
    restore_memory_profile_recursive_max_depth: int | None


def current_rss_mb() -> float | None:
    """Return current process RSS in MB when available."""
    return _pipeline_current_rss_mb()


def log_morpion_checkpoint_memory_phase(
    phase: str,
    *,
    path: str | Path | None = None,
    nodes: int | None = None,
    generation: int | None = None,
) -> None:
    """Log one lightweight current-RSS checkpoint memory marker."""
    parts = [
        f"phase={phase}",
        f"rss_mb={_metric_value(current_rss_mb())}",
    ]
    if nodes is not None:
        parts.append(f"nodes={nodes}")
    if generation is not None:
        parts.append(f"generation={generation}")
    if path is not None:
        parts.append(f"path={path}")
    LOGGER.info("[memory] %s", " ".join(parts))


@dataclass(slots=True)
class RestoreMemoryLogger:
    """Opt-in checkpoint restore RSS and object-size phase logger."""

    checkpoint_path: Path
    compressed_checkpoint_bytes: int | None
    recursive_enabled: bool = False
    recursive_max_objects: int | None = None
    recursive_max_depth: int | None = None
    started_at: float = field(default_factory=time.perf_counter)

    def callback(self, phase: str, metadata: Mapping[str, object]) -> None:
        """Receive one Anemone restore phase callback."""
        self.log(phase, **cast("dict[str, Any]", dict(metadata)))

    def log(
        self,
        phase: str,
        *,
        raw_payload: object | None = None,
        typed_payload: SearchRuntimeCheckpointPayload | None = None,
        raw_checkpoint_referenced: bool | None = None,
        typed_checkpoint_referenced: bool | None = None,
        **metadata: object,
    ) -> None:
        """Emit one structured restore-memory phase line."""
        node_count = _restore_metadata_value(metadata, "node_count", "nodes")
        if node_count is None:
            node_count = _raw_checkpoint_node_count(raw_payload)
        if node_count is None and typed_payload is not None:
            node_count = len(typed_payload.tree.nodes)

        branch_count = _restore_metadata_value(metadata, "branch_count", "branches")
        if branch_count is None and typed_payload is not None:
            branch_count = getattr(typed_payload.tree, "branch_count", None)

        parts = [
            f"phase={phase}",
            f"rss_mb={_metric_value(current_rss_mb())}",
            f"elapsed_s={_metric_value(time.perf_counter() - self.started_at)}",
            f"path={self.checkpoint_path}",
            f"compressed_checkpoint_bytes={_metric_value(self.compressed_checkpoint_bytes)}",
            f"node_count={_metric_value(node_count)}",
            f"branch_count={_metric_value(branch_count)}",
            f"raw_checkpoint_referenced={_metric_value(raw_checkpoint_referenced)}",
            f"typed_checkpoint_referenced={_metric_value(typed_checkpoint_referenced)}",
        ]
        gc_counts = gc.get_count()
        parts.extend(
            [
                f"gc_count0={gc_counts[0]}",
                f"gc_count1={gc_counts[1]}",
                f"gc_count2={gc_counts[2]}",
            ]
        )
        raw_recursive_mb = self._recursive_size_mb(raw_payload)
        if raw_recursive_mb is not None:
            parts.append(f"raw_decoded_recursive_mb={_metric_value(raw_recursive_mb)}")
        typed_recursive_mb = self._recursive_size_mb(typed_payload)
        if typed_recursive_mb is not None:
            parts.append(
                f"typed_checkpoint_payload_recursive_mb={_metric_value(typed_recursive_mb)}"
            )
        for key, value in metadata.items():
            if key in {
                "node_count",
                "nodes",
                "branch_count",
                "branches",
            }:
                continue
            parts.append(f"{key}={_metric_value(value)}")
        LOGGER.info("[restore-memory] %s", " ".join(parts))

    def _recursive_size_mb(self, value: object | None) -> float | None:
        if not self.recursive_enabled or value is None:
            return None

        stats = DeepSizeStats(max_objects=self.recursive_max_objects)
        byte_count = deep_size(
            value,
            seen=set(),
            max_depth=self.recursive_max_depth,
            stats=stats,
        )
        return byte_count / (1024 * 1024)


def restore_memory_logger_for_checkpoint_path(
    checkpoint_path: Path,
    *,
    enabled: bool,
    recursive_enabled: bool = False,
    recursive_max_objects: int | None = None,
    recursive_max_depth: int | None = None,
) -> RestoreMemoryLogger | None:
    """Build the opt-in checkpoint restore-memory logger for a path."""
    if not enabled:
        return None
    checkpoint_bytes = _checkpoint_artifact_bytes(checkpoint_path)
    return RestoreMemoryLogger(
        checkpoint_path=checkpoint_path,
        compressed_checkpoint_bytes=checkpoint_bytes,
        recursive_enabled=recursive_enabled,
        recursive_max_objects=recursive_max_objects,
        recursive_max_depth=recursive_max_depth,
    )


def _restore_memory_logger_for_path(
    args: _RestoreMemoryArgs,
    checkpoint_path: Path,
) -> RestoreMemoryLogger | None:
    return restore_memory_logger_for_checkpoint_path(
        checkpoint_path,
        enabled=args.restore_memory_profile,
        recursive_enabled=args.restore_memory_profile_recursive,
        recursive_max_objects=args.restore_memory_profile_recursive_max_objects,
        recursive_max_depth=args.restore_memory_profile_recursive_max_depth,
    )


def _restore_metadata_value(
    metadata: Mapping[str, object],
    *keys: str,
) -> object | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _raw_checkpoint_node_count(raw_payload: object | None) -> int | None:
    if not isinstance(raw_payload, Mapping):
        return None
    raw_tree = raw_payload.get("tree")
    if not isinstance(raw_tree, Mapping):
        return None
    raw_nodes = raw_tree.get("nodes")
    if not isinstance(raw_nodes, list):
        return None
    return len(raw_nodes)
