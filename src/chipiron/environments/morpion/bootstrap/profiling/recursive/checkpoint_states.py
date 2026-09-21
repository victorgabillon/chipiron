"""Checkpoint-backed state diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .deep_size import (
    DeepSizeStats,
    deep_size,
    deep_size_stats_capped,
    size_or_zero,
)
from .object_access import (
    qualified_type_name,
    raw_getattr,
    raw_getattr_present,
    safe_object_dict,
)
from .tree_topology import tree_node_slot

if TYPE_CHECKING:
    from .context import CheckpointPayloadStore

LOGGER = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_HANDLE_SCAN_CAP = 50_000
_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_DEFAULT_PAYLOAD_SHAPE_TOP_N = 8
_DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS = 4
_ANCHOR_PAYLOAD_TYPE_SUFFIX = "AnchorCheckpointStatePayload"
_DELTA_PAYLOAD_TYPE_SUFFIX = "DeltaCheckpointStatePayload"
_KNOWN_CHECKPOINT_STORE_ATTR_NAMES = (
    "state_payloads_by_node_id",
    "payloads",
    "_state_payloads_by_node_id",
    "_payloads",
    "payloads_by_node_id",
    "_payloads_by_node_id",
)

__all__ = [
    "checkpoint_state_histograms",
    "checkpoint_state_roots_detail_histogram",
]


@dataclass(slots=True)
class _CheckpointResolverHandleStats:
    """Checkpoint-handle diagnostics grouped by resolver identity."""

    resolver_type: str
    checkpoint_handle_count: int = 0
    materialized_handle_count: int = 0
    unmaterialized_handle_count: int = 0
    referenced_payload_keys_by_mapping_id: dict[int, set[object]] = field(
        default_factory=dict
    )


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


def _shallow_object_and_dict_size(value: object, *, seen: set[int]) -> int:
    total = 0
    value_id = id(value)
    if value_id not in seen:
        seen.add(value_id)
        total += size_or_zero(value)
    raw_dict = safe_object_dict(value)
    if raw_dict is not None:
        dict_id = id(raw_dict)
        if dict_id not in seen:
            seen.add(dict_id)
            total += size_or_zero(raw_dict)
    return total


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


def _exclusive_shell_size(roots: Iterable[object | None], *, seen: set[int]) -> int:
    total = 0
    for root in roots:
        if root is None:
            continue
        total += _shallow_object_and_dict_size(root, seen=seen)
    return total


def _checkpoint_payload_kind(value: object) -> str | None:
    payload_type = qualified_type_name(value)
    if payload_type.endswith(_ANCHOR_PAYLOAD_TYPE_SUFFIX):
        return "anchor"
    if payload_type.endswith(_DELTA_PAYLOAD_TYPE_SUFFIX):
        return "delta"
    return None


def _handle_resolver(handle: object) -> object | None:
    """Return a checkpoint resolver from known handle layouts."""
    for attr_name in (
        "resolver",
        "_resolver",
        "state_resolver",
        "_state_resolver",
        "checkpoint_state_resolver",
        "_checkpoint_state_resolver",
    ):
        value = raw_getattr(handle, attr_name)
        if value is not None:
            return value
    return None


def _handle_node_id(handle: object) -> object | None:
    """Return the raw checkpoint node id from known handle layouts."""
    for attr_name in ("node_id", "_node_id"):
        value = raw_getattr(handle, attr_name)
        if value is not None:
            return value
    return None


def _is_checkpoint_backed_state_handle(handle: object) -> bool:
    """Return whether one raw handle looks like an Anemone checkpoint handle."""
    return type(handle).__qualname__.endswith("CheckpointBackedStateHandle")


def _resolver_payloads(resolver: object) -> Mapping[object, object] | None:
    """Return checkpoint payload storage from known resolver layouts."""
    _, mapping = _resolver_payload_mapping_details(resolver)
    if isinstance(mapping, Mapping):
        return mapping
    return None


def _resolver_payload_mapping_details(
    resolver: object,
) -> tuple[str | None, Mapping[object, object] | None]:
    """Return payload mapping details from likely raw resolver/owner attrs."""
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        value = raw_getattr(resolver, attr_name)
        if isinstance(value, Mapping):
            return attr_name, value
    owner = raw_getattr(resolver, "owner")
    if owner is None:
        return None, None
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        value = raw_getattr(owner, attr_name)
        if isinstance(value, Mapping):
            return f"owner.{attr_name}", value
    return None, None


def _resolver_resolved_states(resolver: object) -> Mapping[object, object] | None:
    """Return materialized-state storage from likely raw resolver attrs."""
    for attr_name in (
        "_resolved_states",
        "resolved_states",
        "_state_by_node_id",
        "state_by_node_id",
    ):
        value = raw_getattr(resolver, attr_name)
        if isinstance(value, Mapping):
            return value
    return None


def _handle_has_materialized_state(handle: object, resolver: object | None) -> bool:
    """Return whether one raw checkpoint handle already points to materialized state."""
    for attr_name in ("state_", "_state", "_materialized_state"):
        present, value = raw_getattr_present(handle, attr_name)
        if present and value is not None:
            return True
    node_id = _handle_node_id(handle)
    if resolver is None or not isinstance(node_id, int):
        return False
    resolved_states = _resolver_resolved_states(resolver)
    return isinstance(resolved_states, Mapping) and node_id in resolved_states


def _checkpoint_handle_scan_cap(
    nodes: Iterable[object],
    *,
    max_objects: int | None,
    checkpoint_max_handles: int | None,
) -> int | None:
    """Return the maximum checkpoint handles to inspect for one diagnostic pass."""
    if checkpoint_max_handles is not None:
        return max(0, checkpoint_max_handles)
    if max_objects is None:
        return None
    if isinstance(nodes, Sized):
        return min(len(nodes), _DEFAULT_CHECKPOINT_HANDLE_SCAN_CAP)
    return _DEFAULT_CHECKPOINT_HANDLE_SCAN_CAP


def checkpoint_state_histograms(
    nodes: Iterable[object],
    checkpoint_payload_stores: Iterable[CheckpointPayloadStore] = (),
    *,
    max_objects: int | None = None,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    checkpoint_max_handles: int | None = None,
) -> dict[str, object]:
    """Return checkpoint payload/resolver diagnostics without resolving states."""
    handle_type_counts = Counter[str]()
    resolver_ids: set[int] = set()
    payload_ids: set[int] = set()
    resolved_state_ids: set[int] = set()
    anchor_count = 0
    delta_count = 0
    materialized_state_count = 0
    payload_recursive_seen: set[int] = set()
    payload_recursive_bytes = 0
    payload_stats: DeepSizeStats = DeepSizeStats(max_objects=max_objects)
    resolved_recursive_seen: set[int] = set()
    resolved_recursive_bytes = 0
    resolved_stats: DeepSizeStats = DeepSizeStats(max_objects=max_objects)
    payload_store_count = 0
    handles_seen = 0
    payloads_seen = 0
    anchors_seen = 0
    deltas_seen = 0
    materialized_states_seen = 0
    max_handles = _checkpoint_handle_scan_cap(
        nodes,
        max_objects=max_objects,
        checkpoint_max_handles=checkpoint_max_handles,
    )
    handle_scan_capped = False

    def add_payload(payload: object) -> None:
        nonlocal anchor_count
        nonlocal anchors_seen
        nonlocal delta_count
        nonlocal deltas_seen
        nonlocal payload_recursive_bytes
        nonlocal payloads_seen

        if (
            payload_stats.max_objects is not None
            and payload_stats.visited_objects >= payload_stats.max_objects
        ):
            payload_stats.capped = True
            return
        if id(payload) in payload_ids:
            return
        payload_kind = _checkpoint_payload_kind(payload)
        if payload_kind is None:
            return
        payload_ids.add(id(payload))
        payloads_seen += 1
        if payload_kind == "anchor":
            anchor_count += 1
            anchors_seen += 1
        else:
            delta_count += 1
            deltas_seen += 1
        if not payload_stats.capped:
            payload_recursive_bytes += deep_size(
                payload,
                seen=payload_recursive_seen,
                max_depth=max_depth,
                stats=payload_stats,
            )

    def add_materialized_state(state: object) -> None:
        nonlocal materialized_state_count
        nonlocal materialized_states_seen
        nonlocal resolved_recursive_bytes

        materialized_state_count += 1
        materialized_states_seen += 1
        resolved_state_ids.add(id(state))
        if not resolved_stats.capped:
            resolved_recursive_bytes += deep_size(
                state,
                seen=resolved_recursive_seen,
                max_depth=max_depth,
                stats=resolved_stats,
            )

    for payload_store in checkpoint_payload_stores:
        if deep_size_stats_capped(payload_stats):
            break
        payload_store_count += 1
        for payload in payload_store.payloads.values():
            if deep_size_stats_capped(payload_stats):
                break
            add_payload(payload)

    for node in nodes:
        if max_handles is not None and handles_seen >= max_handles:
            handle_scan_capped = True
            break
        handle = tree_node_slot(node, "state_handle_")
        if handle is None:
            continue
        handles_seen += 1
        handle_type_counts[qualified_type_name(handle)] += 1

        state_value = raw_getattr(handle, "state_")
        if state_value is not None:
            add_materialized_state(state_value)

        resolver = _handle_resolver(handle)
        if resolver is None:
            continue
        resolver_ids.add(id(resolver))
        node_id = _handle_node_id(handle)
        payloads = _resolver_payloads(resolver)
        if (
            not payload_stats.capped
            and isinstance(node_id, int)
            and isinstance(payloads, Mapping)
        ):
            payload = payloads.get(node_id)
            if payload is not None:
                add_payload(payload)
        resolved_states = _resolver_resolved_states(resolver)
        if isinstance(resolved_states, Mapping):
            for state in resolved_states.values():
                if payload_stats.capped or resolved_stats.capped:
                    break
                if id(state) in resolved_state_ids:
                    continue
                add_materialized_state(state)

    capped = payload_stats.capped or resolved_stats.capped or handle_scan_capped
    LOGGER.info(
        "[growth-recursive-profile] checkpoint_state_histograms "
        "handles_seen=%s handles_scanned_cap_reached=%s payloads_seen=%s "
        "anchors_seen=%s deltas_seen=%s materialized_states_seen=%s capped=%s",
        handles_seen,
        handle_scan_capped,
        payloads_seen,
        anchors_seen,
        deltas_seen,
        materialized_states_seen,
        capped,
    )

    return {
        "handle_types": dict(handle_type_counts),
        "resolver_count": len(resolver_ids),
        "payload_store_count": payload_store_count,
        "anchor_payload_count": anchor_count,
        "delta_payload_count": delta_count,
        "payload_recursive_bytes": payload_recursive_bytes,
        "materialized_state_count": materialized_state_count,
        "materialized_state_recursive_bytes": resolved_recursive_bytes,
        "handles_seen": handles_seen,
        "payloads_seen": payloads_seen,
        "anchors_seen": anchors_seen,
        "deltas_seen": deltas_seen,
        "materialized_states_seen": materialized_states_seen,
        "max_handles": max_handles,
        "handle_scan_capped": handle_scan_capped,
        "handles_scanned_cap_reached": handle_scan_capped,
        "payload_recursive_visited_objects": payload_stats.visited_objects,
        "materialized_state_recursive_visited_objects": resolved_stats.visited_objects,
        "capped": capped,
    }


def _payload_shape_type_counts(
    values: Iterable[object | None],
    *,
    top_n: int = _DEFAULT_PAYLOAD_SHAPE_TOP_N,
) -> tuple[tuple[str, int], ...]:
    """Return a bounded type census across roots and immediate nested contents."""
    counts = Counter[str]()
    for value in values:
        if value is None:
            continue
        counts[qualified_type_name(value)] += 1
        if isinstance(value, Mapping):
            for key, item in value.items():
                counts[qualified_type_name(key)] += 1
                counts[qualified_type_name(item)] += 1
        elif isinstance(value, list | tuple):
            for item in value[:_DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS]:
                counts[qualified_type_name(item)] += 1
    return tuple(counts.most_common(top_n))


def _payload_shape_memory_stats(
    roots: Iterable[object | None],
    *,
    max_depth: int | None,
    max_objects: int | None,
) -> dict[str, object]:
    """Return shallow and recursive memory stats for one payload-field group."""
    normalized_roots = tuple(root for root in roots if root is not None)
    recursive_stats = DeepSizeStats(max_objects=max_objects)
    recursive_bytes = _exclusive_deep_size(
        normalized_roots,
        seen=set(),
        max_depth=max_depth,
        stats=recursive_stats,
    )
    shallow_bytes = _exclusive_shell_size(normalized_roots, seen=set())
    return {
        "count": len(normalized_roots),
        "shallow_bytes": shallow_bytes,
        "recursive_bytes": recursive_bytes,
        "recursive_visited_objects": recursive_stats.visited_objects,
        "recursive_capped": recursive_stats.capped,
    }


def checkpoint_state_roots_detail_histogram(
    checkpoint_payload_stores: Iterable[CheckpointPayloadStore] = (),
    *,
    max_objects: int | None = None,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
) -> dict[str, object]:
    """Return compact payload-store diagnostics for checkpoint state roots."""
    stores = tuple(checkpoint_payload_stores)
    if not stores:
        return {"present": False}

    store_type_counts = Counter[str]()
    payload_type_counts = Counter[str]()
    payload_store_roots: list[object] = []
    anchor_payloads: list[object] = []
    delta_payloads: list[object] = []
    anchor_refs: list[object] = []
    delta_refs: list[object] = []
    state_summaries: list[object] = []
    state_parent_node_ids: list[object] = []
    payload_count = 0

    for payload_store in stores:
        store = payload_store.payloads
        payload_store_roots.append(store)
        store_type_counts[qualified_type_name(store)] += 1
        for payload in store.values():
            payload_count += 1
            payload_type_counts[qualified_type_name(payload)] += 1
            payload_kind = _checkpoint_payload_kind(payload)
            state_summary = raw_getattr(payload, "state_summary")
            if state_summary is not None:
                state_summaries.append(state_summary)
            if payload_kind == "anchor":
                anchor_payloads.append(payload)
                anchor_ref = raw_getattr(payload, "anchor_ref")
                if anchor_ref is not None:
                    anchor_refs.append(anchor_ref)
            elif payload_kind == "delta":
                delta_payloads.append(payload)
                delta_ref = raw_getattr(payload, "delta_ref")
                if delta_ref is not None:
                    delta_refs.append(delta_ref)
                state_parent_node_id = raw_getattr(payload, "state_parent_node_id")
                if state_parent_node_id is not None:
                    state_parent_node_ids.append(state_parent_node_id)

    store_stats = DeepSizeStats(max_objects=max_objects)
    payload_store_total_bytes = _exclusive_deep_size(
        payload_store_roots,
        seen=set(),
        max_depth=max_depth,
        stats=store_stats,
    )
    anchor_payload_stats = _payload_shape_memory_stats(
        anchor_payloads,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    delta_payload_stats = _payload_shape_memory_stats(
        delta_payloads,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    anchor_ref_stats = _payload_shape_memory_stats(
        anchor_refs,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    delta_ref_stats = _payload_shape_memory_stats(
        delta_refs,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    state_summary_stats = _payload_shape_memory_stats(
        state_summaries,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    state_parent_node_id_stats = _payload_shape_memory_stats(
        state_parent_node_ids,
        max_depth=max_depth,
        max_objects=max_objects,
    )

    return {
        "present": True,
        "payload_store_total_bytes": payload_store_total_bytes,
        "payload_store_total_visited_objects": store_stats.visited_objects,
        "payload_store_total_capped": store_stats.capped,
        "anchor_payload_bytes": anchor_payload_stats["recursive_bytes"],
        "delta_payload_bytes": delta_payload_stats["recursive_bytes"],
        "anchor_ref_bytes": anchor_ref_stats["recursive_bytes"],
        "delta_ref_bytes": delta_ref_stats["recursive_bytes"],
        "state_summary_bytes": state_summary_stats["recursive_bytes"],
        "state_parent_node_id_bytes": state_parent_node_id_stats["recursive_bytes"],
        "payload_store_type_counts": dict(_ordered_counter_items(store_type_counts)),
        "payload_store_kind_counts": {
            "Dense": sum(
                count
                for type_name, count in store_type_counts.items()
                if "Dense" in type_name
            ),
            "Dict": sum(
                count
                for type_name, count in store_type_counts.items()
                if type_name == "dict" or type_name.endswith(".dict")
            ),
            "Other": sum(
                count
                for type_name, count in store_type_counts.items()
                if "Dense" not in type_name
                and type_name != "dict"
                and not type_name.endswith(".dict")
            ),
        },
        "payload_store_count": len(stores),
        "payload_count": payload_count,
        "anchor_payload_count": len(anchor_payloads),
        "delta_payload_count": len(delta_payloads),
        "top_python_types": dict(_ordered_counter_items(payload_type_counts)),
        "anchor_ref_top_python_types": _payload_shape_type_counts(anchor_refs),
        "delta_ref_top_python_types": _payload_shape_type_counts(delta_refs),
        "state_summary_top_python_types": _payload_shape_type_counts(state_summaries),
    }
