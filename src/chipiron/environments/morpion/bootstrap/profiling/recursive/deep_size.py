"""Generic recursive deep-size traversal for Morpion profiling."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

from .object_access import (
    ATOMIC_TYPES,
    CONTAINER_TYPES,
    iter_object_attribute_values,
    should_skip_deep,
)

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64

__all__ = [
    "DeepSizeStats",
    "deep_size",
    "deep_size_stats_capped",
    "mb",
    "measure_standalone_reachable",
    "size_or_zero",
]


@dataclass(slots=True)
class DeepSizeStats:
    """Mutable counters for one recursive-size traversal."""

    visited_objects: int = 0
    max_objects: int | None = None
    capped: bool = False
    max_depth_reached_count: int = 0
    recursion_error_count: int = 0


def mb(byte_count: int) -> float:
    """Convert a byte count to mebibytes."""
    return byte_count / (1024 * 1024)


def size_or_zero(value: object) -> int:
    """Return ``sys.getsizeof(value)``, or zero when sizing is unsupported."""
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def measure_standalone_reachable(
    value: object,
    *,
    max_depth: int | None,
    max_objects: int | None,
) -> tuple[int, DeepSizeStats]:
    """Measure one value with a fresh recursive traversal state."""
    stats = DeepSizeStats(max_objects=max_objects)
    byte_count = deep_size(value, seen=set(), max_depth=max_depth, stats=stats)
    return byte_count, stats


def deep_size(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
    stats: DeepSizeStats | None = None,
) -> int:
    """Return recursive size while avoiding cycles and lazy properties.

    Traversal is intentionally limited to builtin containers, actual
    ``__dict__`` mappings, and declared ``__slots__``. It does not inspect
    ``dir(obj)`` and therefore avoids calling materializing runtime properties.
    """
    active_stats = stats
    if active_stats is None:
        active_stats = DeepSizeStats(max_objects=max_objects)
    elif max_objects is not None:
        active_stats.max_objects = max_objects
    return _deep_size(obj, seen=seen, max_depth=max_depth, depth=0, stats=active_stats)


def _mark_recursion_error(stats: DeepSizeStats) -> None:
    stats.recursion_error_count += 1
    stats.capped = True


def deep_size_stats_capped(stats: DeepSizeStats) -> bool:
    """Return whether a recursive-size traversal has reached a cap."""
    return stats.capped


def _try_push_deep_size_object(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None,
    depth: int,
    stack: list[tuple[object, int]],
    stats: DeepSizeStats,
) -> int:
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    if stats.max_objects is not None and stats.visited_objects >= stats.max_objects:
        stats.capped = True
        return 0

    seen.add(obj_id)
    stats.visited_objects += 1
    size = size_or_zero(obj)

    if isinstance(obj, ATOMIC_TYPES) or should_skip_deep(obj):
        return size
    if max_depth is not None and depth >= max_depth:
        stats.max_depth_reached_count += 1
        stats.capped = True
        return size

    stack.append((obj, depth))
    return size


def _deep_size(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None,
    depth: int,
    stats: DeepSizeStats,
) -> int:
    stack: list[tuple[object, int]] = []
    total_size = _try_push_deep_size_object(
        obj,
        seen=seen,
        max_depth=max_depth,
        depth=depth,
        stack=stack,
        stats=stats,
    )

    while stack:
        current, current_depth = stack.pop()
        next_depth = current_depth + 1

        if isinstance(current, Mapping):
            try:
                for key, value in current.items():
                    total_size += _try_push_deep_size_object(
                        key,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
                    total_size += _try_push_deep_size_object(
                        value,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
            except RecursionError:
                _mark_recursion_error(stats)
            continue

        if isinstance(current, CONTAINER_TYPES):
            try:
                for item in current:
                    total_size += _try_push_deep_size_object(
                        item,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
            except RecursionError:
                _mark_recursion_error(stats)
            continue

        try:
            for attr_value in iter_object_attribute_values(current):
                total_size += _try_push_deep_size_object(
                    attr_value,
                    seen=seen,
                    max_depth=max_depth,
                    depth=next_depth,
                    stack=stack,
                    stats=stats,
                )
        except RecursionError:
            _mark_recursion_error(stats)

    return total_size
