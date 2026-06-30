"""Checkpoint I/O helpers for Morpion bootstrap runtimes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    SearchRuntimeCheckpointPayload,
    read_sharded_checkpoint_manifest,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckpointIoMetrics:
    """Compact checkpoint I/O metrics for stable structured logging."""

    path: str
    bytes: int | None = None
    file_format: str | None = None
    encoder: str | None = None
    payload_build_s: float | None = None
    jsonable_s: float | None = None
    json_encode_s: float | None = None
    compress_s: float | None = None
    write_s: float | None = None
    json_load_s: float | None = None
    payload_decode_s: float | None = None
    runtime_rebuild_s: float | None = None
    total_s: float | None = None
    uncompressed_bytes: int | None = None
    compression_ratio: float | None = None
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    node_count: int | None = None
    anchor_count: int | None = None
    delta_count: int | None = None
    cache: str | None = None
    runtime_checkpoint_format: str | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpointPayloadCacheEntry:
    """Decoded checkpoint payload retained briefly between validation and restore."""

    path: Path
    bytes: int
    mtime_ns: int
    payload: SearchRuntimeCheckpointPayload


@dataclass(slots=True)
class _ValidatedCheckpointPayloadCache:
    """Mutable holder for the short-lived validated checkpoint payload."""

    entry: _ValidatedCheckpointPayloadCacheEntry | None = None


_validated_checkpoint_payload_cache = _ValidatedCheckpointPayloadCache()


def checkpoint_io_metrics_to_dict(metrics: CheckpointIoMetrics) -> dict[str, object]:
    """Return JSON-friendly checkpoint I/O metrics."""
    return {
        "path": metrics.path,
        "bytes": metrics.bytes,
        "format": metrics.file_format,
        "encoder": metrics.encoder,
        "payload_build_s": metrics.payload_build_s,
        "jsonable_s": metrics.jsonable_s,
        "json_encode_s": metrics.json_encode_s,
        "compress_s": metrics.compress_s,
        "write_s": metrics.write_s,
        "json_load_s": metrics.json_load_s,
        "payload_decode_s": metrics.payload_decode_s,
        "runtime_rebuild_s": metrics.runtime_rebuild_s,
        "total_s": metrics.total_s,
        "uncompressed_bytes": metrics.uncompressed_bytes,
        "compression_ratio": metrics.compression_ratio,
        "rss_before_mb": metrics.rss_before_mb,
        "rss_after_mb": metrics.rss_after_mb,
        "node_count": metrics.node_count,
        "anchor_count": metrics.anchor_count,
        "delta_count": metrics.delta_count,
        "cache": metrics.cache,
        "runtime_checkpoint_format": metrics.runtime_checkpoint_format,
    }


def _is_sharded_runtime_checkpoint_path(path: str | Path) -> bool:
    """Return whether a path points to a sharded runtime checkpoint directory."""
    resolved_path = Path(path)
    return resolved_path.is_dir() and (resolved_path / "manifest.json").is_file()


def _checkpoint_artifact_bytes(path: str | Path) -> int | None:
    """Return compressed bytes for a file or best-effort shard bytes for a directory."""
    resolved_path = Path(path)
    try:
        if resolved_path.is_file():
            return resolved_path.stat().st_size
        if _is_sharded_runtime_checkpoint_path(resolved_path):
            manifest = read_sharded_checkpoint_manifest(resolved_path / "manifest.json")
            shard_bytes = sum(
                shard.compressed_bytes or 0
                for shard in manifest.shards
                if shard.compressed_bytes is not None
            )
            return (resolved_path / "manifest.json").stat().st_size + shard_bytes
    except OSError:
        return None
    return None


def _checkpoint_node_counts(
    payload: SearchRuntimeCheckpointPayload,
) -> tuple[int, int, int]:
    """Return total, anchor, and delta node counts for one checkpoint payload."""
    nodes = payload.tree.nodes
    anchor_count = sum(
        1
        for node_payload in nodes
        if isinstance(node_payload.state_payload, AnchorCheckpointStatePayload)
    )
    delta_count = sum(
        1
        for node_payload in nodes
        if isinstance(node_payload.state_payload, DeltaCheckpointStatePayload)
    )
    return len(nodes), anchor_count, delta_count


def _metric_value(value: object) -> str:
    """Render one metric field as a stable log token."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _log_checkpoint_metrics(operation: str, metrics: CheckpointIoMetrics) -> None:
    """Emit one stable checkpoint metrics log line."""
    parts = [
        f"operation={operation}",
        f"path={metrics.path}",
        f"bytes={_metric_value(metrics.bytes)}",
        f"nodes={_metric_value(metrics.node_count)}",
        f"anchors={_metric_value(metrics.anchor_count)}",
        f"deltas={_metric_value(metrics.delta_count)}",
    ]
    if metrics.file_format is not None:
        parts.append(f"format={metrics.file_format}")
    if metrics.encoder is not None:
        parts.append(f"encoder={metrics.encoder}")
    if metrics.cache is not None:
        parts.append(f"cache={metrics.cache}")
    if metrics.runtime_checkpoint_format is not None:
        parts.append(f"runtime_checkpoint_format={metrics.runtime_checkpoint_format}")
    parts.extend(
        [
            f"payload_build_s={_metric_value(metrics.payload_build_s)}",
            f"jsonable_s={_metric_value(metrics.jsonable_s)}",
            f"json_encode_s={_metric_value(metrics.json_encode_s)}",
            f"compress_s={_metric_value(metrics.compress_s)}",
            f"write_s={_metric_value(metrics.write_s)}",
            f"json_load_s={_metric_value(metrics.json_load_s)}",
            f"payload_decode_s={_metric_value(metrics.payload_decode_s)}",
            f"runtime_rebuild_s={_metric_value(metrics.runtime_rebuild_s)}",
            f"total_s={_metric_value(metrics.total_s)}",
            f"uncompressed_bytes={_metric_value(metrics.uncompressed_bytes)}",
            f"compression_ratio={_metric_value(metrics.compression_ratio)}",
            f"rss_before_mb={_metric_value(metrics.rss_before_mb)}",
            f"rss_after_mb={_metric_value(metrics.rss_after_mb)}",
        ]
    )
    LOGGER.info("[checkpoint-metrics] %s", " ".join(parts))


def _checkpoint_payload_cache_identity(path: str | Path) -> tuple[Path, int, int]:
    """Return the identity fields that make a cached payload safe to reuse."""
    resolved_path = Path(path).resolve()
    path_stat = resolved_path.stat()
    return resolved_path, path_stat.st_size, path_stat.st_mtime_ns


def cache_morpion_search_checkpoint_payload_for_restore(
    path: str | Path,
    payload: SearchRuntimeCheckpointPayload,
) -> None:
    """Retain one validated payload for an immediately following restore."""
    try:
        resolved_path, bytes_loaded, mtime_ns = _checkpoint_payload_cache_identity(path)
    except FileNotFoundError:
        _validated_checkpoint_payload_cache.entry = None
        return
    _validated_checkpoint_payload_cache.entry = _ValidatedCheckpointPayloadCacheEntry(
        path=resolved_path,
        bytes=bytes_loaded,
        mtime_ns=mtime_ns,
        payload=payload,
    )


def _pop_cached_morpion_search_checkpoint_payload_for_restore(
    path: str | Path,
) -> tuple[SearchRuntimeCheckpointPayload, int] | None:
    """Return and clear the matching validated payload cache entry, if any."""
    entry = _validated_checkpoint_payload_cache.entry
    if entry is None:
        return None
    try:
        resolved_path, bytes_loaded, mtime_ns = _checkpoint_payload_cache_identity(path)
    except FileNotFoundError:
        _validated_checkpoint_payload_cache.entry = None
        return None
    if (
        entry.path != resolved_path
        or entry.bytes != bytes_loaded
        or entry.mtime_ns != mtime_ns
    ):
        _validated_checkpoint_payload_cache.entry = None
        return None
    _validated_checkpoint_payload_cache.entry = None
    return entry.payload, entry.bytes
