"""Checkpoint payload diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from .checkpoint_states import (
    _checkpoint_handle_scan_cap,
    _CheckpointResolverHandleStats,
    _handle_has_materialized_state,
    _handle_node_id,
    _handle_resolver,
    _is_checkpoint_backed_state_handle,
    _resolver_payloads,
)
from .deep_size import DeepSizeStats, deep_size, size_or_zero
from .object_access import (
    qualified_type_name,
    raw_getattr,
    safe_object_dict,
)
from .tree_topology import tree_node_slot

if TYPE_CHECKING:
    from .context import CheckpointPayloadStore, ComponentProfileRecord

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_DEFAULT_PAYLOAD_SHAPE_TOP_N = 8
_DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS = 4
_ANCHOR_PAYLOAD_TYPE_SUFFIX = "AnchorCheckpointStatePayload"
_DELTA_PAYLOAD_TYPE_SUFFIX = "DeltaCheckpointStatePayload"

__all__ = [
    "checkpoint_payload_lifetime_histograms",
    "checkpoint_payload_shape_histograms",
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


def checkpoint_payload_lifetime_histograms(
    nodes: Iterable[object],
    checkpoint_payload_stores: Iterable[CheckpointPayloadStore] = (),
    *,
    checkpoint_payload_store_records: Iterable[ComponentProfileRecord] = (),
    checkpoint_max_handles: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Return per-resolver checkpoint payload lifetime diagnostics without get()."""
    payload_stores = tuple(checkpoint_payload_stores)
    payload_store_records = tuple(checkpoint_payload_store_records)
    handle_stats_by_resolver_id: dict[int, _CheckpointResolverHandleStats] = {}
    checkpoint_handle_count = 0
    handles_without_resolver_count = 0
    resolver_ids_seen_by_handles: set[int] = set()
    max_handles = _checkpoint_handle_scan_cap(
        nodes,
        max_objects=None,
        checkpoint_max_handles=checkpoint_max_handles,
    )
    handles_seen = 0
    handle_scan_capped = False

    for node in nodes:
        if max_handles is not None and handles_seen >= max_handles:
            handle_scan_capped = True
            break
        handle = tree_node_slot(node, "state_handle_")
        if handle is None:
            continue
        handles_seen += 1
        if not _is_checkpoint_backed_state_handle(handle):
            continue
        checkpoint_handle_count += 1
        resolver = _handle_resolver(handle)
        if resolver is None:
            handles_without_resolver_count += 1
            continue
        resolver_id = id(resolver)
        resolver_ids_seen_by_handles.add(resolver_id)
        stats = handle_stats_by_resolver_id.setdefault(
            resolver_id,
            _CheckpointResolverHandleStats(
                resolver_type=qualified_type_name(resolver),
            ),
        )
        stats.checkpoint_handle_count += 1
        if _handle_has_materialized_state(handle, resolver):
            stats.materialized_handle_count += 1
        else:
            stats.unmaterialized_handle_count += 1
        node_id = _handle_node_id(handle)
        payloads = _resolver_payloads(resolver)
        if (
            isinstance(node_id, int)
            and isinstance(payloads, Mapping)
            and node_id in payloads
        ):
            stats.referenced_payload_keys_by_mapping_id.setdefault(
                id(payloads), set()
            ).add(node_id)

    all_handles_share_one_resolver = (
        checkpoint_handle_count > 0
        and handles_without_resolver_count == 0
        and len(resolver_ids_seen_by_handles) == 1
    )
    records_by_index = dict(enumerate(payload_store_records, start=1))
    histograms: list[dict[str, object]] = []
    seen_store_resolver_ids: set[int] = set()

    for index, payload_store in enumerate(payload_stores, start=1):
        seen_store_resolver_ids.add(payload_store.resolver_id)
        resolver_stats = handle_stats_by_resolver_id.get(payload_store.resolver_id)
        record = records_by_index.get(index)
        referenced_payload_entries_count = 0
        if resolver_stats is not None:
            referenced_payload_entries_count = len(
                resolver_stats.referenced_payload_keys_by_mapping_id.get(
                    id(payload_store.payloads),
                    set(),
                )
            )
        histograms.append({
            "resolver_type": payload_store.resolver_type,
            "resolver_object_id": payload_store.resolver_id,
            "payload_mapping_attr_name": payload_store.attr_name,
            "payload_mapping_type": qualified_type_name(payload_store.payloads),
            "payload_mapping_length": len(payload_store.payloads),
            "anchor_count": payload_store.anchor_count,
            "delta_count": payload_store.delta_count,
            "payload_mapping_recursive_bytes": (0 if record is None else record.bytes),
            "payload_mapping_shallow_bytes": size_or_zero(payload_store.payloads),
            "checkpoint_backed_state_handle_count": (
                0 if resolver_stats is None else resolver_stats.checkpoint_handle_count
            ),
            "materialized_handle_count": (
                0
                if resolver_stats is None
                else resolver_stats.materialized_handle_count
            ),
            "unmaterialized_handle_count": (
                0
                if resolver_stats is None
                else resolver_stats.unmaterialized_handle_count
            ),
            "payload_entries_still_referenced_count": referenced_payload_entries_count,
            "all_handles_share_one_resolver": all_handles_share_one_resolver,
            "handle_scan_capped": handle_scan_capped,
        })

    for resolver_id, resolver_stats in handle_stats_by_resolver_id.items():
        if resolver_id in seen_store_resolver_ids:
            continue
        histograms.append({
            "resolver_type": resolver_stats.resolver_type,
            "resolver_object_id": resolver_id,
            "payload_mapping_attr_name": None,
            "payload_mapping_type": None,
            "payload_mapping_length": 0,
            "anchor_count": 0,
            "delta_count": 0,
            "payload_mapping_recursive_bytes": 0,
            "payload_mapping_shallow_bytes": 0,
            "checkpoint_backed_state_handle_count": resolver_stats.checkpoint_handle_count,
            "materialized_handle_count": resolver_stats.materialized_handle_count,
            "unmaterialized_handle_count": resolver_stats.unmaterialized_handle_count,
            "payload_entries_still_referenced_count": 0,
            "all_handles_share_one_resolver": all_handles_share_one_resolver,
            "handle_scan_capped": handle_scan_capped,
        })

    if histograms:
        return tuple(histograms)
    return (
        {
            "present": False,
            "all_handles_share_one_resolver": False,
            "handle_scan_capped": handle_scan_capped,
        },
    )


def _small_payload_sample(
    value: object,
    *,
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = _DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS,
) -> object:
    """Return a short, repr-safe structural sample for one payload field."""
    if depth >= max_depth:
        return qualified_type_name(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:80] + ("..." if len(value) > 80 else "")
    if isinstance(value, Mapping):
        sampled_items: dict[object, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                sampled_items["..."] = f"+{len(value) - max_items} more"
                break
            sampled_items[key] = _small_payload_sample(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        return sampled_items
    if isinstance(value, list | tuple):
        sampled_sequence: list[object] = [
            _small_payload_sample(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            sampled_sequence.append(f"... +{len(value) - max_items} more")
        return tuple(sampled_sequence) if isinstance(value, tuple) else sampled_sequence
    if isinstance(value, set | frozenset):
        sampled_set: list[object] = [
            _small_payload_sample(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            sampled_set.append(f"... +{len(value) - max_items} more")
        return sampled_set
    return qualified_type_name(value)


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


def _payload_shape_dict_key_counts(
    values: Iterable[object | None],
    *,
    top_n: int = _DEFAULT_PAYLOAD_SHAPE_TOP_N,
) -> tuple[tuple[str, int], ...]:
    """Return the most common immediate dict keys across one payload field."""
    counts = Counter[str]()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in value:
            counts[key if isinstance(key, str) else repr(key)] += 1
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


def checkpoint_payload_shape_histograms(
    checkpoint_payload_stores: Iterable[CheckpointPayloadStore] = (),
    *,
    max_objects: int | None = None,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
) -> tuple[dict[str, object], ...]:
    """Return read-only payload field shape diagnostics for checkpoint stores."""
    histograms: list[dict[str, object]] = []
    for payload_store in checkpoint_payload_stores:
        payloads = tuple(payload_store.payloads.values())
        anchor_payloads = [
            payload
            for payload in payloads
            if _checkpoint_payload_kind(payload) == "anchor"
        ]
        delta_payloads = [
            payload
            for payload in payloads
            if _checkpoint_payload_kind(payload) == "delta"
        ]
        anchor_refs = [
            raw_getattr(payload, "anchor_ref") for payload in anchor_payloads
        ]
        delta_refs = [raw_getattr(payload, "delta_ref") for payload in delta_payloads]
        state_summaries = [
            raw_getattr(payload, "state_summary") for payload in payloads
        ]
        state_parent_branches = [
            raw_getattr(payload, "state_parent_branch") for payload in delta_payloads
        ]
        state_parent_node_ids = [
            raw_getattr(payload, "state_parent_node_id") for payload in delta_payloads
        ]
        payload_type_counts = Counter[str](
            qualified_type_name(payload) for payload in payloads
        )

        histograms.append({
            "resolver_type": payload_store.resolver_type,
            "resolver_object_id": payload_store.resolver_id,
            "payload_store_type": qualified_type_name(payload_store.payloads),
            "payload_store_length": len(payload_store.payloads),
            "total_payload_count": len(payloads),
            "anchor_payload_count": len(anchor_payloads),
            "delta_payload_count": len(delta_payloads),
            "payload_type_counts": dict(_ordered_counter_items(payload_type_counts)),
            "anchor_payload_objects": _payload_shape_memory_stats(
                anchor_payloads,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "delta_payload_objects": _payload_shape_memory_stats(
                delta_payloads,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "anchor_ref": _payload_shape_memory_stats(
                anchor_refs,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "delta_ref": _payload_shape_memory_stats(
                delta_refs,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "state_summary": _payload_shape_memory_stats(
                state_summaries,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "state_parent_branch": _payload_shape_memory_stats(
                state_parent_branches,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "state_parent_node_id": _payload_shape_memory_stats(
                state_parent_node_ids,
                max_depth=max_depth,
                max_objects=max_objects,
            ),
            "anchor_ref_top_python_types": _payload_shape_type_counts(anchor_refs),
            "delta_ref_top_python_types": _payload_shape_type_counts(delta_refs),
            "state_summary_top_python_types": _payload_shape_type_counts(
                state_summaries
            ),
            "state_parent_branch_top_python_types": _payload_shape_type_counts(
                state_parent_branches
            ),
            "anchor_ref_common_dict_keys": _payload_shape_dict_key_counts(anchor_refs),
            "delta_ref_common_dict_keys": _payload_shape_dict_key_counts(delta_refs),
            "state_summary_common_dict_keys": _payload_shape_dict_key_counts(
                state_summaries
            ),
            "state_parent_branch_common_dict_keys": _payload_shape_dict_key_counts(
                state_parent_branches
            ),
            "anchor_ref_sample": _small_payload_sample(anchor_refs[0])
            if anchor_refs
            else None,
            "delta_ref_sample": _small_payload_sample(delta_refs[0])
            if delta_refs
            else None,
            "state_summary_sample": (
                _small_payload_sample(state_summaries[0]) if state_summaries else None
            ),
        })
    if histograms:
        return tuple(histograms)
    return ({"present": False},)
