"""Node-evaluation diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from chipiron.environments.morpion.bootstrap.pipeline_memory import format_metric

from .deep_size import DeepSizeStats, deep_size, size_or_zero
from .object_access import (
    CONTAINER_TYPES,
    iter_direct_field_entries,
    len_or_none,
    qualified_type_name,
    raw_getattr,
    small_len_bucket,
)
from .tree_topology import node_tree_evaluation

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_NODE_EVALUATION_VALUE_SLOTS = (
    "direct_value",
    "_backed_up_value",
)
_NODE_EVALUATION_RUNTIME_SLOTS = (
    "decision_ordering_",
    "pv_state_",
    "branch_frontier_",
    "backup_runtime_",
)
_RUNTIME_STATE_SLOT_LABELS: dict[str, str] = {
    "decision_ordering_": "DecisionOrderingState",
    "pv_state_": "PrincipalVariationState",
    "branch_frontier_": "BranchFrontierState",
    "backup_runtime_": "Top2ExactnessPvRuntime",
}

__all__ = [
    "node_evaluation_runtime_detail_histograms",
    "node_evaluation_runtime_histograms",
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


def node_evaluation_runtime_histograms(nodes: Iterable[object]) -> dict[str, object]:
    """Return materialized NodeMaxEvaluation runtime-state counters."""
    counts = Counter[str]()
    eval_type_counts = Counter[str]()
    for node in nodes:
        node_eval = node_tree_evaluation(node)
        if node_eval is None:
            continue
        eval_type_counts[qualified_type_name(node_eval)] += 1
        for slot_name in _NODE_EVALUATION_RUNTIME_SLOTS:
            if raw_getattr(node_eval, slot_name) is not None:
                counts[f"{slot_name}_non_none"] += 1
        for slot_name in _NODE_EVALUATION_VALUE_SLOTS:
            if raw_getattr(node_eval, slot_name) is not None:
                counts[f"{slot_name}_non_none"] += 1
    return {
        "node_evaluation_types": dict(eval_type_counts),
        "runtime_state_counts": dict(counts),
    }


def node_evaluation_runtime_detail_histograms(
    nodes: Iterable[object],
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Return runtime-state size and field diagnostics grouped by state class."""
    node_count = 0
    states_by_label: dict[str, list[object]] = {}

    for node in nodes:
        node_count += 1
        node_eval = node_tree_evaluation(node)
        if node_eval is None:
            continue
        for slot_name in _NODE_EVALUATION_RUNTIME_SLOTS:
            runtime_state = raw_getattr(node_eval, slot_name)
            if runtime_state is None:
                continue
            label = _RUNTIME_STATE_SLOT_LABELS.get(
                slot_name,
                type(runtime_state).__qualname__,
            )
            states_by_label.setdefault(label, []).append(runtime_state)

    if not states_by_label:
        return ({"present": False, "node_count": node_count},)

    histograms: list[dict[str, object]] = []
    for label, states in sorted(states_by_label.items()):
        state_type_counts = Counter[str]()
        empty_count = 0
        non_empty_count = 0
        field_type_counts = Counter[str]()
        field_len_buckets = Counter[str]()
        field_shallow_bytes = Counter[str]()
        top_child_type_counts = Counter[str]()

        for state in states:
            state_type_counts[qualified_type_name(state)] += 1
            state_is_empty = True
            for field_name, field_value in iter_direct_field_entries(state):
                field_type_counts[
                    f"{field_name}:{qualified_type_name(field_value)}"
                ] += 1
                field_shallow_bytes[field_name] += size_or_zero(field_value)
                field_len = len_or_none(field_value)
                if field_len is not None:
                    field_len_buckets[
                        f"{field_name}:{small_len_bucket(field_len)}"
                    ] += 1
                    if field_len > 0:
                        state_is_empty = False
                elif field_value not in (None, False, 0):
                    state_is_empty = False

                if isinstance(field_value, Mapping):
                    top_child_type_counts.update(
                        qualified_type_name(item) for item in field_value.values()
                    )
                elif isinstance(field_value, CONTAINER_TYPES):
                    top_child_type_counts.update(
                        qualified_type_name(item) for item in field_value
                    )
                elif field_value is not None:
                    top_child_type_counts[qualified_type_name(field_value)] += 1

            if state_is_empty:
                empty_count += 1
            else:
                non_empty_count += 1

        states_stats = DeepSizeStats(max_objects=max_objects)
        recursive_bytes = _exclusive_deep_size(
            states,
            seen=set(),
            max_depth=max_depth,
            stats=states_stats,
        )
        field_recursive_bytes: dict[str, int] = {}
        for field_name in sorted({
            name
            for state in states
            for name, _value in iter_direct_field_entries(state)
        }):
            field_values = [
                field_value
                for state in states
                for name, field_value in iter_direct_field_entries(state)
                if name == field_name
            ]
            field_stats = DeepSizeStats(max_objects=max_objects)
            field_recursive_bytes[field_name] = _exclusive_deep_size(
                field_values,
                seen=set(),
                max_depth=max_depth,
                stats=field_stats,
            )

        histograms.append({
            "present": True,
            "runtime_state_label": label,
            "node_count": node_count,
            "state_count": len(states),
            "state_types": dict(_ordered_counter_items(state_type_counts)),
            "recursive_reachable_bytes": recursive_bytes,
            "recursive_reachable_capped": states_stats.capped,
            "average_recursive_bytes_per_state": format_metric(
                recursive_bytes / len(states)
            ),
            "average_recursive_bytes_per_node": format_metric(
                recursive_bytes / node_count if node_count else None
            ),
            "empty_state_count": empty_count,
            "non_empty_state_count": non_empty_count,
            "field_type_counts": dict(_ordered_counter_items(field_type_counts)),
            "field_len_buckets": dict(_ordered_counter_items(field_len_buckets)),
            "field_shallow_bytes": dict(_ordered_counter_items(field_shallow_bytes)),
            "field_recursive_bytes": field_recursive_bytes,
            "top_child_object_types": dict(top_child_type_counts.most_common(10)),
        })

    return tuple(histograms)
