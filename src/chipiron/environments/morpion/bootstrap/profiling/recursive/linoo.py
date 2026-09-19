"""Linoo selector diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sized
from enum import Enum

from chipiron.environments.morpion.bootstrap.pipeline_memory import format_metric

from .context import find_linoo_selector_root
from .deep_size import (
    DeepSizeStats,
    deep_size,
    measure_standalone_reachable,
    size_or_zero,
)
from .object_access import (
    ATOMIC_TYPES,
    CONTAINER_TYPES,
    CONTAINER_VALUE_TYPES,
    iter_direct_field_entries,
    iter_object_attribute_values,
    qualified_type_name,
    raw_getattr,
    raw_getattr_present,
    safe_call_no_args,
    should_skip_deep,
    slot_names,
)
from .tree_topology import node_id_or_none

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_DEFAULT_LINOO_NODE_STATE_SAMPLE_CAP = 5_000
_LINOO_NODE_STATE_TABLE_ATTR_NAME = "_node_state_by_id"
_LINOO_SLOT_VALUE_KIND_ORDER = (
    "AlgorithmNode",
    "TreeNode",
    "int",
    "str",
    "enum",
    "tuple",
    "list",
    "dict",
    "set",
    "None",
)

__all__ = [
    "linoo_candidate_heap_histogram",
    "linoo_deep_breakdown_histograms",
    "linoo_node_state_slots_histogram",
    "linoo_node_state_table_histogram",
    "linoo_selector_detail_histogram",
    "linoo_state_histograms",
    "resolve_linoo_selector",
    "selector_node_status_from_table",
]


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


def _exclusive_deep_size(
    roots: Iterable[object | None],
    *,
    seen: set[int],
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    stats: DeepSizeStats,
) -> int:
    total = 0
    for root in roots:
        if root is None:
            continue
        total += deep_size(root, seen=seen, max_depth=max_depth, stats=stats)
    return total


def resolve_linoo_selector(
    selector: object | None,
) -> tuple[object | None, Mapping[object, object] | None]:
    """Return the concrete Linoo selector plus its node-state table when present."""
    if selector is None:
        return None, None
    linoo_selector = find_linoo_selector_root(selector)
    if linoo_selector is None:
        return None, None
    node_state_by_id = raw_getattr(linoo_selector, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
    if not isinstance(node_state_by_id, Mapping):
        return linoo_selector, None
    return linoo_selector, node_state_by_id


def selector_node_status_from_table(
    node_state_by_id: Mapping[object, object] | None,
    node: object,
) -> str | None:
    """Return the Linoo selector status for a node id when available."""
    if node_state_by_id is None:
        return None
    node_id = node_id_or_none(node)
    if node_id is None:
        return None
    state = node_state_by_id.get(node_id)
    if state is None:
        return "opened"
    status = raw_getattr(state, "status")
    return status if isinstance(status, str) else str(status)


def _linoo_slot_value_kind(value: object) -> str:
    if value is None:
        return "None"
    type_name = qualified_type_name(value)
    if type_name.endswith("AlgorithmNode"):
        return "AlgorithmNode"
    if type_name.endswith("TreeNode"):
        return "TreeNode"
    if isinstance(value, Enum):
        return "enum"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, set | frozenset):
        return "set"
    return type_name


def _object_reaches_type_suffix(
    root: object,
    suffixes: tuple[str, ...],
    *,
    max_objects: int | None = 50_000,
) -> bool:
    """Return whether ``root`` reaches an object whose type name has a suffix."""
    seen: set[int] = set()
    stack: list[object] = [root]
    visited = 0
    while stack:
        value = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        visited += 1
        if max_objects is not None and visited > max_objects:
            return False
        type_name = qualified_type_name(value)
        if any(type_name.endswith(suffix) for suffix in suffixes):
            return True
        if isinstance(value, ATOMIC_TYPES) or should_skip_deep(value):
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                stack.append(key)
                stack.append(item)
            continue
        if isinstance(value, CONTAINER_TYPES):
            stack.extend(value)
            continue
        stack.extend(iter_object_attribute_values(value))
    return False


def _tree_reachability_flags(root: object) -> dict[str, bool]:
    return {
        "reaches_algorithm_node": _object_reaches_type_suffix(
            root,
            ("AlgorithmNode",),
        ),
        "reaches_tree_node": _object_reaches_type_suffix(root, ("TreeNode",)),
    }


def linoo_state_histograms(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
) -> dict[str, object]:
    """Return sparse Linoo state-table diagnostics when a Linoo selector is present."""
    if selector is None:
        return {"present": False}
    linoo_selector, node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None:
        raw_node_state_by_id = raw_getattr(selector, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
        return {
            "present": True,
            "selector_type": qualified_type_name(selector),
            "node_state_table_type": qualified_type_name(raw_node_state_by_id),
        }
    assert isinstance(node_state_by_id, Mapping)

    default_count = 0
    non_default_count = 0
    state_type_counts = Counter[str]()
    slot_value_type_counts = Counter[str]()
    node_state_table_shallow_bytes = size_or_zero(node_state_by_id)
    container_shallow_total = node_state_table_shallow_bytes
    container_recursive_seen: set[int] = set()
    container_recursive_total = 0

    for state in node_state_by_id.values():
        state_type_counts[qualified_type_name(state)] += 1
        is_default = raw_getattr(state, "is_default")
        if callable(is_default):
            if bool(safe_call_no_args(is_default)):
                default_count += 1
            else:
                non_default_count += 1
        for slot_name in slot_names(state):
            slot_value = raw_getattr(state, slot_name)
            if slot_value is None:
                continue
            slot_value_type_counts[qualified_type_name(slot_value)] += 1
            if isinstance(slot_value, CONTAINER_VALUE_TYPES):
                container_shallow_total += size_or_zero(slot_value)
                container_recursive_total += deep_size(
                    slot_value,
                    seen=container_recursive_seen,
                    max_depth=max_depth,
                )

    return {
        "present": True,
        "selector_type": qualified_type_name(linoo_selector),
        "node_state_count": len(node_state_by_id),
        "default_count": default_count,
        "non_default_count": non_default_count,
        "node_state_table_type": qualified_type_name(node_state_by_id),
        "state_types": dict(state_type_counts),
        "slot_value_types": dict(slot_value_type_counts),
        "node_state_table_shallow_bytes": node_state_table_shallow_bytes,
        "container_shallow_bytes": container_shallow_total,
        "container_recursive_bytes": container_recursive_total,
        **_tree_reachability_flags(node_state_by_id),
    }


def linoo_deep_breakdown_histograms(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Return standalone reachable-size diagnostics for direct Linoo fields."""
    linoo_selector, _node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None:
        return ({"present": False},)

    breakdowns: list[dict[str, object]] = []
    for field_name, field_value in iter_direct_field_entries(linoo_selector):
        recursive_reachable_bytes, stats = measure_standalone_reachable(
            field_value,
            max_depth=max_depth,
            max_objects=max_objects,
        )
        breakdowns.append({
            "present": True,
            "selector_type": qualified_type_name(linoo_selector),
            "field_name": field_name,
            "value_type": qualified_type_name(field_value),
            "shallow_bytes": size_or_zero(field_value),
            "recursive_reachable_bytes": recursive_reachable_bytes,
            "visited_objects": stats.visited_objects,
            "capped": stats.capped,
            "max_depth_reached_count": stats.max_depth_reached_count,
            "recursion_error_count": stats.recursion_error_count,
        })
    if not breakdowns:
        return (
            {
                "present": True,
                "selector_type": qualified_type_name(linoo_selector),
                "field_count": 0,
            },
        )
    return tuple(breakdowns)


def _iter_linoo_heap_entries(candidates_by_depth: object) -> Iterator[object]:
    if not isinstance(candidates_by_depth, Mapping):
        return
    for heap in candidates_by_depth.values():
        if isinstance(heap, Iterable) and not isinstance(heap, str | bytes | bytearray):
            yield from heap


def _linoo_candidate_stale_count(
    linoo_selector: object,
    candidates_by_depth: object,
) -> int | None:
    versions = raw_getattr(linoo_selector, "_candidate_versions_by_node_id")
    present = raw_getattr(linoo_selector, "_candidate_heap_present_by_node_id")
    if not isinstance(versions, Mapping) or not isinstance(present, Mapping):
        return None

    stale_count = 0
    for entry in _iter_linoo_heap_entries(candidates_by_depth):
        if not isinstance(entry, tuple) or len(entry) < 3:
            continue
        raw_node_id = entry[1]
        raw_version = entry[2]
        if not isinstance(raw_node_id, int) or not isinstance(raw_version, int):
            continue
        if not bool(present.get(raw_node_id, False)):
            stale_count += 1
            continue
        current_version = versions.get(raw_node_id)
        if isinstance(current_version, int) and current_version != raw_version:
            stale_count += 1
    return stale_count


def linoo_candidate_heap_histogram(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> dict[str, object]:
    """Return focused diagnostics for Linoo candidate heap storage."""
    linoo_selector, _node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None:
        return {"present": False}

    candidates_by_depth = raw_getattr(linoo_selector, "_candidates_by_depth")
    if not isinstance(candidates_by_depth, Mapping):
        return {
            "present": True,
            "selector_type": qualified_type_name(linoo_selector),
            "candidate_heap_table_type": qualified_type_name(candidates_by_depth),
            **_tree_reachability_flags(candidates_by_depth),
        }

    heap_type_counts = Counter[str]()
    entry_type_counts = Counter[str]()
    entry_shape_counts = Counter[str]()
    entry_value_type_counts = Counter[str]()
    heap_count = 0
    candidate_entry_count = 0
    heap_shallow_bytes = size_or_zero(candidates_by_depth)

    for heap in candidates_by_depth.values():
        heap_count += 1
        heap_type_counts[qualified_type_name(heap)] += 1
        heap_shallow_bytes += size_or_zero(heap)
        if not isinstance(heap, Iterable) or isinstance(heap, str | bytes | bytearray):
            continue
        for entry in heap:
            candidate_entry_count += 1
            entry_type_counts[qualified_type_name(entry)] += 1
            if isinstance(entry, tuple):
                entry_shape_counts[f"tuple[{len(entry)}]"] += 1
                for item in entry:
                    entry_value_type_counts[qualified_type_name(item)] += 1
            else:
                entry_shape_counts[qualified_type_name(entry)] += 1

    recursive_bytes, stats = measure_standalone_reachable(
        candidates_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    stale_count = _linoo_candidate_stale_count(linoo_selector, candidates_by_depth)

    return {
        "present": True,
        "selector_type": qualified_type_name(linoo_selector),
        "candidate_depth_count": len(candidates_by_depth),
        "candidate_heap_count": heap_count,
        "candidate_entry_count": candidate_entry_count,
        "candidate_stale_entry_count": stale_count,
        "candidate_stale_fraction": (
            None
            if stale_count is None or candidate_entry_count == 0
            else format_metric(stale_count / candidate_entry_count)
        ),
        "candidate_heap_table_shallow_bytes": size_or_zero(candidates_by_depth),
        "candidate_heaps_shallow_bytes": heap_shallow_bytes,
        "candidate_heaps_recursive_reachable_bytes": recursive_bytes,
        "candidate_heaps_recursive_reachable_visited_objects": stats.visited_objects,
        "candidate_heaps_recursive_reachable_capped": stats.capped,
        "candidate_heaps_recursive_reachable_max_depth_reached_count": (
            stats.max_depth_reached_count
        ),
        "candidate_heaps_recursive_reachable_recursion_error_count": (
            stats.recursion_error_count
        ),
        "candidate_heap_types": dict(_ordered_counter_items(heap_type_counts)),
        "candidate_entry_types": dict(_ordered_counter_items(entry_type_counts)),
        "candidate_entry_shapes": dict(_ordered_counter_items(entry_shape_counts)),
        "candidate_entry_value_types": dict(
            _ordered_counter_items(entry_value_type_counts)
        ),
        **_tree_reachability_flags(candidates_by_depth),
    }


def linoo_selector_detail_histogram(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> dict[str, object]:
    """Return compact C3a diagnostics for the largest Linoo live structures."""
    linoo_selector, node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None:
        return {"present": False}

    selector_recursive_bytes, selector_stats = measure_standalone_reachable(
        linoo_selector,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    node_state_table = node_state_by_id if node_state_by_id is not None else {}
    table_recursive_bytes, table_stats = measure_standalone_reachable(
        node_state_table,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    node_state_object_shallow_total = 0
    status_breakdown = Counter[str]()
    for state in node_state_table.values():
        node_state_object_shallow_total += size_or_zero(state)
        status = raw_getattr(state, "status")
        status_breakdown[str(status)] += 1

    candidates_by_depth = raw_getattr(linoo_selector, "_candidates_by_depth")
    candidate_heap = linoo_candidate_heap_histogram(
        linoo_selector,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    depth_stats_by_depth = raw_getattr(linoo_selector, "_depth_stats_by_depth")
    depth_stats_recursive_bytes, depth_stats_recursive = measure_standalone_reachable(
        depth_stats_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    frontier_ids_by_depth = raw_getattr(linoo_selector, "_frontier_node_ids_by_depth")
    frontier_recursive_bytes, frontier_stats = measure_standalone_reachable(
        frontier_ids_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )

    return {
        "present": True,
        "selector_type": qualified_type_name(linoo_selector),
        "total_selector_recursive_bytes": selector_recursive_bytes,
        "total_selector_recursive_visited_objects": selector_stats.visited_objects,
        "total_selector_recursive_capped": selector_stats.capped,
        "node_state_table_attr_name": _LINOO_NODE_STATE_TABLE_ATTR_NAME,
        "node_state_table_type": qualified_type_name(node_state_table),
        "node_state_table_shallow_bytes": size_or_zero(node_state_table),
        "node_state_table_recursive_bytes": table_recursive_bytes,
        "node_state_table_recursive_visited_objects": table_stats.visited_objects,
        "node_state_table_recursive_capped": table_stats.capped,
        "node_state_count": len(node_state_table),
        "node_state_object_shallow_total_bytes": node_state_object_shallow_total,
        "status_representation_breakdown": dict(
            _ordered_counter_items(status_breakdown)
        ),
        "candidates_by_depth_type": qualified_type_name(candidates_by_depth),
        "candidate_heap_count": candidate_heap.get("candidate_heap_count"),
        "total_candidate_entries": candidate_heap.get("candidate_entry_count"),
        "stale_candidate_entries": candidate_heap.get("candidate_stale_entry_count"),
        "candidate_tuple_shape_type_breakdown": candidate_heap.get(
            "candidate_entry_shapes"
        ),
        "candidate_entry_types": candidate_heap.get("candidate_entry_types"),
        "candidate_entry_value_types": candidate_heap.get(
            "candidate_entry_value_types"
        ),
        "candidates_by_depth_recursive_bytes": candidate_heap.get(
            "candidate_heaps_recursive_reachable_bytes"
        ),
        "depth_stats_type": qualified_type_name(depth_stats_by_depth),
        "depth_stats_count": (
            len(depth_stats_by_depth)
            if isinstance(depth_stats_by_depth, Sized)
            else None
        ),
        "depth_stats_shallow_bytes": size_or_zero(depth_stats_by_depth),
        "depth_stats_recursive_bytes": depth_stats_recursive_bytes,
        "depth_stats_recursive_visited_objects": depth_stats_recursive.visited_objects,
        "depth_stats_recursive_capped": depth_stats_recursive.capped,
        "frontier_ids_by_depth_type": qualified_type_name(frontier_ids_by_depth),
        "frontier_ids_by_depth_shallow_bytes": size_or_zero(frontier_ids_by_depth),
        "frontier_ids_by_depth_recursive_bytes": frontier_recursive_bytes,
        "frontier_ids_by_depth_recursive_visited_objects": (
            frontier_stats.visited_objects
        ),
        "frontier_ids_by_depth_recursive_capped": frontier_stats.capped,
        **_tree_reachability_flags(linoo_selector),
    }


def linoo_node_state_table_histogram(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> dict[str, object]:
    """Return recursive and shallow diagnostics for the Linoo node-state table."""
    linoo_selector, node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None or node_state_by_id is None:
        return {"present": False}

    key_type_counts = Counter[str]()
    value_type_counts = Counter[str]()
    node_states_shallow_bytes = 0
    node_state_count = 0
    for key, value in node_state_by_id.items():
        key_type_counts[qualified_type_name(key)] += 1
        value_type_counts[qualified_type_name(value)] += 1
        node_states_shallow_bytes += size_or_zero(value)
        node_state_count += 1

    table_recursive_reachable_bytes, table_stats = measure_standalone_reachable(
        node_state_by_id,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    key_stats = DeepSizeStats(max_objects=max_objects)
    keys_recursive_reachable_bytes = _exclusive_deep_size(
        node_state_by_id.keys(),
        seen=set(),
        max_depth=max_depth,
        stats=key_stats,
    )
    value_stats = DeepSizeStats(max_objects=max_objects)
    values_recursive_reachable_bytes = _exclusive_deep_size(
        node_state_by_id.values(),
        seen=set(),
        max_depth=max_depth,
        stats=value_stats,
    )
    node_state_stats = DeepSizeStats(max_objects=max_objects)
    node_states_recursive_reachable_bytes = _exclusive_deep_size(
        node_state_by_id.values(),
        seen=set(),
        max_depth=max_depth,
        stats=node_state_stats,
    )

    return {
        "present": True,
        "selector_type": qualified_type_name(linoo_selector),
        "table_attr_name": _LINOO_NODE_STATE_TABLE_ATTR_NAME,
        "table_type": qualified_type_name(node_state_by_id),
        "table_length": len(node_state_by_id),
        "table_shallow_bytes": size_or_zero(node_state_by_id),
        "key_type_counts": dict(_ordered_counter_items(key_type_counts)),
        "value_type_counts": dict(_ordered_counter_items(value_type_counts)),
        "table_recursive_reachable_bytes": table_recursive_reachable_bytes,
        "table_recursive_reachable_visited_objects": table_stats.visited_objects,
        "table_recursive_reachable_capped": table_stats.capped,
        "table_recursive_reachable_max_depth_reached_count": (
            table_stats.max_depth_reached_count
        ),
        "table_recursive_reachable_recursion_error_count": (
            table_stats.recursion_error_count
        ),
        "keys_recursive_reachable_bytes": keys_recursive_reachable_bytes,
        "keys_recursive_reachable_visited_objects": key_stats.visited_objects,
        "keys_recursive_reachable_capped": key_stats.capped,
        "keys_recursive_reachable_max_depth_reached_count": (
            key_stats.max_depth_reached_count
        ),
        "keys_recursive_reachable_recursion_error_count": (
            key_stats.recursion_error_count
        ),
        "values_recursive_reachable_bytes": values_recursive_reachable_bytes,
        "values_recursive_reachable_visited_objects": value_stats.visited_objects,
        "values_recursive_reachable_capped": value_stats.capped,
        "values_recursive_reachable_max_depth_reached_count": (
            value_stats.max_depth_reached_count
        ),
        "values_recursive_reachable_recursion_error_count": (
            value_stats.recursion_error_count
        ),
        "node_state_count": node_state_count,
        "node_states_shallow_bytes": node_states_shallow_bytes,
        "node_states_recursive_reachable_bytes": (
            node_states_recursive_reachable_bytes
        ),
        "node_states_recursive_reachable_visited_objects": (
            node_state_stats.visited_objects
        ),
        "node_states_recursive_reachable_capped": node_state_stats.capped,
        "node_states_recursive_reachable_max_depth_reached_count": (
            node_state_stats.max_depth_reached_count
        ),
        "node_states_recursive_reachable_recursion_error_count": (
            node_state_stats.recursion_error_count
        ),
        **_tree_reachability_flags(node_state_by_id),
    }


def linoo_node_state_slots_histogram(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
    sample_cap: int = _DEFAULT_LINOO_NODE_STATE_SAMPLE_CAP,
) -> dict[str, object]:
    """Return sampled slot-level diagnostics for Linoo node states."""
    linoo_selector, node_state_by_id = resolve_linoo_selector(selector)
    if linoo_selector is None or node_state_by_id is None:
        return {"present": False}

    sampled_states: list[object] = []
    for state in node_state_by_id.values():
        if len(sampled_states) >= sample_cap:
            break
        sampled_states.append(state)

    if not sampled_states:
        return {
            "present": True,
            "selector_type": qualified_type_name(linoo_selector),
            "sampled_state_count": 0,
            "sample_cap": sample_cap,
        }

    slot_names_seen: list[str] = []
    slot_names_set: set[str] = set()
    slot_value_type_counts = Counter[str]()
    slot_value_kind_counts = Counter[str]()
    slot_observation_counts = Counter[str]()
    slot_shallow_bytes = Counter[str]()
    slot_values_by_name: dict[str, list[object]] = {}

    for state in sampled_states:
        for slot_name in slot_names(state):
            if slot_name not in slot_names_set:
                slot_names_set.add(slot_name)
                slot_names_seen.append(slot_name)
            present, slot_value = raw_getattr_present(state, slot_name)
            if not present:
                continue
            slot_value_type_counts[qualified_type_name(slot_value)] += 1
            slot_value_kind_counts[_linoo_slot_value_kind(slot_value)] += 1
            slot_observation_counts[slot_name] += 1
            slot_shallow_bytes[slot_name] += size_or_zero(slot_value)
            slot_values_by_name.setdefault(slot_name, []).append(slot_value)

    slot_recursive_reachable_bytes: dict[str, int] = {}
    slot_recursive_reachable_capped: dict[str, bool] = {}
    slot_recursive_reachable_max_depth_reached_count: dict[str, int] = {}
    slot_recursive_reachable_recursion_error_count: dict[str, int] = {}
    for slot_name in slot_names_seen:
        values = slot_values_by_name.get(slot_name, [])
        stats = DeepSizeStats(max_objects=max_objects)
        slot_recursive_reachable_bytes[slot_name] = _exclusive_deep_size(
            values,
            seen=set(),
            max_depth=max_depth,
            stats=stats,
        )
        slot_recursive_reachable_capped[slot_name] = stats.capped
        slot_recursive_reachable_max_depth_reached_count[slot_name] = (
            stats.max_depth_reached_count
        )
        slot_recursive_reachable_recursion_error_count[slot_name] = (
            stats.recursion_error_count
        )

    sampled_states_recursive_stats = DeepSizeStats(max_objects=max_objects)
    sampled_states_recursive_reachable_bytes = _exclusive_deep_size(
        sampled_states,
        seen=set(),
        max_depth=max_depth,
        stats=sampled_states_recursive_stats,
    )

    return {
        "present": True,
        "selector_type": qualified_type_name(linoo_selector),
        "sample_cap": sample_cap,
        "sampled_state_count": len(sampled_states),
        "slot_names": tuple(slot_names_seen),
        "slot_value_type_counts": dict(_ordered_counter_items(slot_value_type_counts)),
        "slot_value_kind_counts": dict(
            _ordered_counter_items(
                slot_value_kind_counts, order=_LINOO_SLOT_VALUE_KIND_ORDER
            )
        ),
        "slot_observation_counts": dict(
            _ordered_counter_items(slot_observation_counts, order=slot_names_seen)
        ),
        "slot_average_shallow_bytes": {
            slot_name: format_metric(
                slot_shallow_bytes[slot_name] / slot_observation_counts[slot_name]
            )
            for slot_name in slot_names_seen
            if slot_observation_counts[slot_name] > 0
        },
        "slot_recursive_reachable_bytes": {
            slot_name: slot_recursive_reachable_bytes[slot_name]
            for slot_name in slot_names_seen
        },
        "slot_recursive_reachable_capped": {
            slot_name: slot_recursive_reachable_capped[slot_name]
            for slot_name in slot_names_seen
        },
        "slot_recursive_reachable_max_depth_reached_count": {
            slot_name: slot_recursive_reachable_max_depth_reached_count[slot_name]
            for slot_name in slot_names_seen
        },
        "slot_recursive_reachable_recursion_error_count": {
            slot_name: slot_recursive_reachable_recursion_error_count[slot_name]
            for slot_name in slot_names_seen
        },
        "sampled_states_recursive_reachable_bytes": (
            sampled_states_recursive_reachable_bytes
        ),
        "sampled_states_recursive_reachable_capped": sampled_states_recursive_stats.capped,
        "sampled_states_recursive_reachable_max_depth_reached_count": (
            sampled_states_recursive_stats.max_depth_reached_count
        ),
        "sampled_states_recursive_reachable_recursion_error_count": (
            sampled_states_recursive_stats.recursion_error_count
        ),
        "average_recursive_reachable_bytes_per_state": format_metric(
            sampled_states_recursive_reachable_bytes / len(sampled_states)
        ),
    }
