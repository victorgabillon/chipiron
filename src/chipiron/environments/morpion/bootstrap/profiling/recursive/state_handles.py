"""State-handle diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from .deep_size import DeepSizeStats, deep_size, size_or_zero
from .linoo import resolve_linoo_selector, selector_node_status_from_table
from .object_access import (
    iter_direct_field_entries,
    qualified_type_name,
    raw_getattr,
    safe_call_no_args,
)
from .tree_topology import (
    branch_ref_count,
    child_link_count_from_storage,
    node_depth_or_none,
    node_eval_bool,
    node_tree_node,
    tree_node_slot,
)

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64

__all__ = [
    "state_eviction_runtime_histogram",
    "state_handle_materialization_detail_histogram",
    "state_handle_storage_kind",
    "state_retention_by_node_status_histogram",
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


def _node_state_handle(node: object) -> object | None:
    handle = tree_node_slot(node, "state_handle_")
    if handle is not None:
        return handle
    tree_node = node_tree_node(node) or node
    return raw_getattr(tree_node, "state_handle_") or raw_getattr(
        tree_node,
        "state_handle",
    )


def _state_from_materialized_handle(handle: object | None) -> object | None:
    if handle is None:
        return None
    return raw_getattr(handle, "state_")


def state_handle_storage_kind(handle: object | None) -> str:
    """Return the known storage kind for a Morpion state handle."""
    type_name = qualified_type_name(handle)
    if type_name.endswith("MaterializedStateHandle"):
        return "MaterializedStateHandle"
    if type_name.endswith("CheckpointBackedStateHandle"):
        return "CheckpointBackedStateHandle"
    if handle is None:
        return "None"
    return "other"


def state_handle_materialization_detail_histogram(
    nodes: Iterable[object],
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> dict[str, object]:
    """Return materialized-state pressure diagnostics without resolving handles."""
    node_count = 0
    handle_type_counts = Counter[str]()
    handle_storage_kind_counts = Counter[str]()
    state_type_counts = Counter[str]()
    field_type_counts = Counter[str]()
    field_shallow_bytes = Counter[str]()
    frozenset_field_counts = Counter[str]()
    frozenset_field_shallow_bytes = Counter[str]()
    materialized_states: list[object] = []
    materialized_state_ids: set[int] = set()
    materialized_state_count = 0
    checkpoint_backed_state_count = 0
    materialized_morpion_state_count = 0

    for node in nodes:
        node_count += 1
        handle = _node_state_handle(node)
        handle_type = qualified_type_name(handle)
        handle_kind = state_handle_storage_kind(handle)
        handle_type_counts[handle_type] += 1
        handle_storage_kind_counts[handle_kind] += 1
        if handle_kind == "CheckpointBackedStateHandle":
            checkpoint_backed_state_count += 1

        state = _state_from_materialized_handle(handle)
        if state is None:
            continue
        materialized_state_count += 1
        state_type = qualified_type_name(state)
        state_type_counts[state_type] += 1
        if state_type == "MorpionState" or state_type.endswith(".MorpionState"):
            materialized_morpion_state_count += 1
        if id(state) not in materialized_state_ids:
            materialized_state_ids.add(id(state))
            materialized_states.append(state)
        for field_name, field_value in iter_direct_field_entries(state):
            field_type_counts[f"{field_name}:{qualified_type_name(field_value)}"] += 1
            field_shallow_bytes[field_name] += size_or_zero(field_value)
            if isinstance(field_value, frozenset):
                frozenset_field_counts[field_name] += 1
                frozenset_field_shallow_bytes[field_name] += size_or_zero(field_value)

    states_stats = DeepSizeStats(max_objects=max_objects)
    materialized_states_recursive_bytes = _exclusive_deep_size(
        materialized_states,
        seen=set(),
        max_depth=max_depth,
        stats=states_stats,
    )

    return {
        "node_count_scanned": node_count,
        "handle_type_counts": dict(_ordered_counter_items(handle_type_counts)),
        "handle_storage_kind_counts": dict(
            _ordered_counter_items(
                handle_storage_kind_counts,
                order=(
                    "MaterializedStateHandle",
                    "CheckpointBackedStateHandle",
                    "other",
                    "None",
                ),
            )
        ),
        "materialized_state_count": materialized_state_count,
        "unique_materialized_state_count": len(materialized_states),
        "checkpoint_backed_state_count": checkpoint_backed_state_count,
        "state_type_counts": dict(_ordered_counter_items(state_type_counts)),
        "materialized_morpion_state_count": materialized_morpion_state_count,
        "materialized_states_recursive_bytes": materialized_states_recursive_bytes,
        "materialized_states_recursive_visited_objects": states_stats.visited_objects,
        "materialized_states_recursive_capped": states_stats.capped,
        "materialized_states_recursive_max_depth_reached_count": (
            states_stats.max_depth_reached_count
        ),
        "materialized_states_recursive_recursion_error_count": (
            states_stats.recursion_error_count
        ),
        "top_materialized_state_field_types": dict(
            _ordered_counter_items(field_type_counts)
        ),
        "materialized_state_field_shallow_bytes": dict(
            _ordered_counter_items(field_shallow_bytes)
        ),
        "materialized_state_frozenset_field_counts": dict(
            _ordered_counter_items(frozenset_field_counts)
        ),
        "materialized_state_frozenset_field_shallow_bytes": dict(
            _ordered_counter_items(frozenset_field_shallow_bytes)
        ),
    }


def state_retention_by_node_status_histogram(
    nodes: Iterable[object],
    *,
    selector: object | None = None,
) -> dict[str, object]:
    """Return materialized-state retention grouped by cheap node status flags."""
    node_count = 0
    materialized_state_count = 0
    materialized_on_internal_nodes = 0
    materialized_on_frontier_nodes = 0
    materialized_on_terminal_nodes = 0
    materialized_on_exact_nodes = 0
    materialized_on_all_branches_generated_nodes = 0
    materialized_on_no_unopened_branch_nodes = 0
    candidate_cold_evictable_materialized_state_count = 0
    depth_counts = Counter[str]()
    materialized_depth_counts = Counter[str]()
    selector_status_counts = Counter[str]()
    materialized_selector_status_counts = Counter[str]()
    _linoo_selector, node_state_by_id = resolve_linoo_selector(selector)
    del _linoo_selector

    for node in nodes:
        node_count += 1
        tree_node = node_tree_node(node) or node
        depth = node_depth_or_none(node)
        depth_bucket = "unknown" if depth is None else str(depth)
        depth_counts[depth_bucket] += 1

        handle = _node_state_handle(node)
        has_materialized_state = _state_from_materialized_handle(handle) is not None
        if has_materialized_state:
            materialized_state_count += 1
            materialized_depth_counts[depth_bucket] += 1

        child_count = child_link_count_from_storage(
            raw_getattr(tree_node, "branches_children_")
        )
        has_children = child_count > 0
        unopened_branch_count = branch_ref_count(
            raw_getattr(tree_node, "non_opened_branches_")
        )
        has_no_unopened_branches = unopened_branch_count == 0
        all_branches_generated = bool(raw_getattr(tree_node, "all_branches_generated"))
        terminal = node_eval_bool(node, "is_terminal")
        exact = node_eval_bool(node, "has_exact_value")
        selector_status = selector_node_status_from_table(node_state_by_id, node)
        selector_status_key = selector_status or "unknown"
        selector_status_counts[selector_status_key] += 1
        is_frontier = selector_status == "frontier"
        is_openable_without_selector = (
            selector_status is None
            and not all_branches_generated
            and not has_no_unopened_branches
        )
        is_openable = is_frontier or is_openable_without_selector

        if not has_materialized_state:
            continue

        materialized_selector_status_counts[selector_status_key] += 1
        if has_children:
            materialized_on_internal_nodes += 1
        if is_frontier:
            materialized_on_frontier_nodes += 1
        if terminal is True:
            materialized_on_terminal_nodes += 1
        if exact is True:
            materialized_on_exact_nodes += 1
        if all_branches_generated:
            materialized_on_all_branches_generated_nodes += 1
        if has_no_unopened_branches:
            materialized_on_no_unopened_branch_nodes += 1

        retained_finished_or_internal = (
            has_children
            or terminal is True
            or exact is True
            or all_branches_generated
            or has_no_unopened_branches
        )
        if retained_finished_or_internal and not is_openable:
            candidate_cold_evictable_materialized_state_count += 1

    return {
        "node_count_scanned": node_count,
        "materialized_state_count": materialized_state_count,
        "materialized_states_on_opened_internal_nodes": (
            materialized_on_internal_nodes
        ),
        "materialized_states_on_frontier_nodes": materialized_on_frontier_nodes,
        "materialized_states_on_terminal_nodes": materialized_on_terminal_nodes,
        "materialized_states_on_exact_nodes": materialized_on_exact_nodes,
        "materialized_states_on_all_branches_generated_nodes": (
            materialized_on_all_branches_generated_nodes
        ),
        "materialized_states_on_no_unopened_branch_nodes": (
            materialized_on_no_unopened_branch_nodes
        ),
        "candidate_cold_evictable_materialized_state_count": (
            candidate_cold_evictable_materialized_state_count
        ),
        "depth_counts": dict(_ordered_counter_items(depth_counts)),
        "materialized_depth_counts": dict(
            _ordered_counter_items(materialized_depth_counts)
        ),
        "selector_status_counts": dict(_ordered_counter_items(selector_status_counts)),
        "materialized_selector_status_counts": dict(
            _ordered_counter_items(materialized_selector_status_counts)
        ),
    }


def state_eviction_runtime_histogram(runner: object) -> dict[str, object]:
    """Return opt-in growth state-eviction counters when exposed by the runner."""
    payload = safe_call_no_args(raw_getattr(runner, "profile_state_eviction_runtime"))
    if isinstance(payload, Mapping):
        return {"present": True, **dict(payload)}
    metrics = raw_getattr(runner, "_state_eviction_metrics")
    payload = safe_call_no_args(raw_getattr(metrics, "snapshot"))
    if isinstance(payload, Mapping):
        return {"present": True, **dict(payload)}
    return {"present": False}
