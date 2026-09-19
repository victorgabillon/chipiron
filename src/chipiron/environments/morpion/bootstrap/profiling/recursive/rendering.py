"""Rendering and logging helpers for recursive memory profiling."""

from __future__ import annotations

import gc
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.bootstrap.pipeline_memory import format_metric

from .deep_size import mb as _mb
from .deep_size import size_or_zero as _size_or_zero
from .frozenset_ownership import (
    direct_frozenset_ownership_summary as _direct_frozenset_ownership_summary,
)
from .object_access import qualified_type_name as _qualified_type_name

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_PROJECT_TYPE_PREFIXES = ("anemone.", "chipiron.", "atomheart.", "valanga.")
_TRACKED_SHALLOW_TYPE_SUFFIXES = (
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "BranchOrderingKey",
    "Value",
    "AlgorithmNode",
    "TreeNode",
    "NodeMaxEvaluation",
    "CheckpointBackedStateHandle",
    "_LinooNodeState",
    "AnchorCheckpointStatePayload",
    "DeltaCheckpointStatePayload",
)


def _format_name_float_pairs(items: Iterable[tuple[str, float]]) -> str:
    return (
        "[" + ",".join(f"{name}:{format_metric(value)}" for name, value in items) + "]"
    )


def _format_name_int_pairs(items: Iterable[tuple[str, int]]) -> str:
    return "[" + ",".join(f"{name}:{value}" for name, value in items) + "]"


def _ordered_counter_items(
    counts: Mapping[str, int],
    *,
    order: Iterable[str] | None = None,
) -> list[tuple[str, int]]:
    if order is not None:
        ordered_items = [
            (name, counts[name]) for name in order if counts.get(name, 0) > 0
        ]
        if ordered_items:
            return ordered_items
    return sorted(counts.items(), key=lambda item: (item[0],))


def gc_shallow_size_summary(*, top_n: int) -> dict[str, object]:
    """Return one cheap process-wide shallow memory summary from GC objects."""
    type_counts = Counter[str]()
    type_bytes = Counter[str]()
    project_type_counts = Counter[str]()
    project_type_bytes = Counter[str]()
    tracked_counts = Counter[str]()
    tracked_bytes = Counter[str]()

    object_count = 0
    total_shallow_bytes = 0
    gc_objects = gc.get_objects()
    for value in gc_objects:
        object_count += 1
        byte_count = _size_or_zero(value)
        total_shallow_bytes += byte_count
        type_name = _qualified_type_name(value)
        type_counts[type_name] += 1
        type_bytes[type_name] += byte_count
        if type_name.startswith(_PROJECT_TYPE_PREFIXES):
            project_type_counts[type_name] += 1
            project_type_bytes[type_name] += byte_count
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES:
            if type_name == suffix or type_name.endswith(f".{suffix}"):
                tracked_counts[suffix] += 1
                tracked_bytes[suffix] += byte_count
                break

    top_by_bytes = type_bytes.most_common(top_n)
    top_by_count = type_counts.most_common(top_n)
    top_project_by_bytes = project_type_bytes.most_common(top_n)
    top_project_by_count = project_type_counts.most_common(top_n)
    tracked_by_bytes = [
        (suffix, tracked_bytes[suffix])
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES
        if tracked_counts[suffix] or tracked_bytes[suffix]
    ]
    tracked_by_count = [
        (suffix, tracked_counts[suffix])
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES
        if tracked_counts[suffix] or tracked_bytes[suffix]
    ]
    frozenset_ownership_summary = _direct_frozenset_ownership_summary(gc_objects)

    return {
        "object_count": object_count,
        "total_shallow_bytes": total_shallow_bytes,
        "top_by_bytes": top_by_bytes,
        "top_by_count": top_by_count,
        "top_project_by_bytes": top_project_by_bytes,
        "top_project_by_count": top_project_by_count,
        "tracked_by_bytes": tracked_by_bytes,
        "tracked_by_count": tracked_by_count,
        "frozenset_ownership": frozenset_ownership_summary,
    }


def log_gc_shallow_size_summary(*, event: str, top_n: int) -> None:
    """Log process-wide shallow-size and frozenset-ownership summaries."""
    start_time = time.perf_counter()
    LOGGER.info(
        "[growth-recursive-profile] event=%s gc_shallow_size_summary_start",
        event,
    )
    summary = gc_shallow_size_summary(top_n=top_n)
    LOGGER.info(
        "[growth-recursive-profile] event=%s gc_shallow_size_summary_done "
        "elapsed_s=%s object_count=%s",
        event,
        format_metric(time.perf_counter() - start_time),
        summary["object_count"],
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s histogram=gc_shallow_sizes "
        "object_count=%s total_shallow_bytes=%s total_shallow_mb=%s "
        "top_by_bytes=%s top_by_count=%s top_project_by_bytes=%s "
        "top_project_by_count=%s tracked_by_bytes=%s tracked_by_count=%s",
        event,
        summary["object_count"],
        summary["total_shallow_bytes"],
        format_metric(_mb(cast("int", summary["total_shallow_bytes"]))),
        _format_name_int_pairs(cast("list[tuple[str, int]]", summary["top_by_bytes"])),
        _format_name_int_pairs(cast("list[tuple[str, int]]", summary["top_by_count"])),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["top_project_by_bytes"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["top_project_by_count"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["tracked_by_bytes"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["tracked_by_count"])
        ),
    )
    frozenset_ownership = cast(
        "dict[str, object]",
        summary["frozenset_ownership"],
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s histogram=frozenset_ownership "
        "total_count=%s total_shallow_mb=%s len_buckets=%s "
        "morpion_state_count=%s morpion_state_field_refs=%s "
        "morpion_state_field_shallow_mb=%s "
        "morpion_state_field_len_buckets=%s",
        event,
        frozenset_ownership["total_count"],
        format_metric(_mb(cast("int", frozenset_ownership["total_shallow_bytes"]))),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", frozenset_ownership["len_buckets"])
        ),
        frozenset_ownership["morpion_state_count"],
        _format_name_int_pairs(
            cast(
                "list[tuple[str, int]]",
                frozenset_ownership["morpion_state_field_refs"],
            )
        ),
        format_metric(
            _mb(cast("int", frozenset_ownership["morpion_state_field_shallow_bytes"]))
        ),
        _format_name_int_pairs(
            cast(
                "list[tuple[str, int]]",
                frozenset_ownership["morpion_state_field_len_buckets"],
            )
        ),
    )


def log_histogram(event: str, name: str, payload: Mapping[str, object]) -> None:
    """Log one recursive profiling histogram payload."""
    formatted_items: list[str] = []
    for key, value in payload.items():
        if key.endswith("_bytes") and isinstance(value, int):
            formatted_items.append(f"{key}={value}")
            formatted_items.append(f"{key[:-6]}_mb={format_metric(_mb(value))}")
        else:
            formatted_items.append(f"{key}={value!r}")
    LOGGER.info(
        "[growth-recursive-profile] event=%s histogram=%s %s",
        event,
        name,
        " ".join(formatted_items),
    )


__all__ = [
    "gc_shallow_size_summary",
    "log_gc_shallow_size_summary",
    "log_histogram",
]
