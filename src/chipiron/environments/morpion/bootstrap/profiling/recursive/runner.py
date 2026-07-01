"""Top-level recursive memory profiling pass for Morpion growth runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    current_rss_mb,
    format_metric,
)

from .checkpoint_payloads import (
    checkpoint_payload_lifetime_histograms,
    checkpoint_payload_shape_histograms,
)
from .checkpoint_states import (
    checkpoint_state_histograms,
    checkpoint_state_roots_detail_histogram,
)
from .context import (
    CheckpointPayloadStore,
    ComponentProfileRecord,
    RecursiveProfileContext,
    build_recursive_profile_context,
)
from .context import component_roots as _component_roots
from .deep_size import DeepSizeStats, deep_size
from .deep_size import mb as _mb
from .deep_size import size_or_zero as _size_or_zero
from .linoo import (
    linoo_candidate_heap_histogram,
    linoo_deep_breakdown_histograms,
    linoo_node_state_slots_histogram,
    linoo_node_state_table_histogram,
    linoo_selector_detail_histogram,
    linoo_state_histograms,
)
from .node_evaluation import (
    node_evaluation_runtime_detail_histograms,
    node_evaluation_runtime_histograms,
)
from .object_access import qualified_type_name as _qualified_type_name
from .object_access import raw_getattr as _raw_getattr
from .object_access import safe_object_dict as _safe_object_dict
from .rendering import _format_name_float_pairs
from .rendering import log_gc_shallow_size_summary as _log_gc_shallow_size_summary
from .rendering import log_histogram as _log_histogram
from .state_handles import (
    state_eviction_runtime_histogram,
    state_handle_materialization_detail_histogram,
    state_retention_by_node_status_histogram,
)
from .tree_topology import (
    child_link_storage_detail_histogram,
    parent_link_storage_histogram,
    tree_topology_histograms,
)
from .tree_topology import iter_child_branch_refs as _iter_child_branch_refs
from .tree_topology import iter_parent_branch_refs as _iter_parent_branch_refs
from .tree_topology import node_eval_slot as _node_eval_slot
from .tree_topology import node_tree_evaluation as _node_tree_evaluation
from .tree_topology import node_tree_node as _node_tree_node
from .tree_topology import tree_node_slot as _tree_node_slot

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

LOGGER = logging.getLogger(__name__)
_OWNER_MODULE = "recursive.runner"

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
_NODE_EVALUATION_MISC_SLOTS = (
    "backup_policy",
    "objective",
)

_DEFAULT_RECURSIVE_PROFILE_MAX_OBJECTS = 3_000_000
_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64

type ProfileRoot = tuple[str, object]


def _node_state_representation(node: object) -> object | None:
    return _raw_getattr(node, "_state_representation")


def _shallow_object_and_dict_size(value: object, *, seen: set[int]) -> int:
    total = 0
    value_id = id(value)
    if value_id not in seen:
        seen.add(value_id)
        total += _size_or_zero(value)
    raw_dict = _safe_object_dict(value)
    if raw_dict is not None:
        dict_id = id(raw_dict)
        if dict_id not in seen:
            seen.add(dict_id)
            total += _size_or_zero(raw_dict)
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


def _log_standalone_components(
    *,
    event: str,
    context: RecursiveProfileContext,
    max_objects: int | None,
    max_depth: int | None,
    complete_map: bool,
) -> list[ComponentProfileRecord]:
    records: list[ComponentProfileRecord] = []
    for component, root in _component_roots(context):
        stats = DeepSizeStats(max_objects=max_objects)
        byte_count = deep_size(root, seen=set(), max_depth=max_depth, stats=stats)
        records.append(
            ComponentProfileRecord(
                component=f"standalone:{component}",
                bytes=byte_count,
                visited_objects=stats.visited_objects,
                capped=stats.capped,
                max_depth_reached_count=stats.max_depth_reached_count,
                recursion_error_count=stats.recursion_error_count,
            )
        )
        LOGGER.info(
            "[growth-recursive-profile] event=%s mode=standalone component=%s "
            "bytes=%s mb=%s visited_objects=%s max_depth=%s complete_map=%s capped=%s "
            "max_depth_reached_count=%s recursion_error_count=%s",
            event,
            component,
            byte_count,
            format_metric(_mb(byte_count)),
            stats.visited_objects,
            max_depth,
            complete_map,
            stats.capped,
            stats.max_depth_reached_count,
            stats.recursion_error_count,
        )
    return records


def _exclusive_components(context: RecursiveProfileContext) -> tuple[ProfileRoot, ...]:
    nodes = context.nodes
    tree_nodes = tuple(
        tree_node for node in nodes if (tree_node := _node_tree_node(node)) is not None
    )
    node_evaluations = tuple(
        node_eval
        for node in nodes
        if (node_eval := _node_tree_evaluation(node)) is not None
    )
    parent_links = tuple(_tree_node_slot(node, "parent_nodes_") for node in nodes)
    child_links = tuple(_tree_node_slot(node, "branches_children_") for node in nodes)
    unopened_links = tuple(
        _tree_node_slot(node, "non_opened_branches_") for node in nodes
    )
    state_handles = tuple(_tree_node_slot(node, "state_handle_") for node in nodes)
    eval_values = tuple(
        _node_eval_slot(node, slot_name)
        for node in nodes
        for slot_name in _NODE_EVALUATION_VALUE_SLOTS
    )
    eval_runtime_states = tuple(
        _node_eval_slot(node, slot_name)
        for node in nodes
        for slot_name in _NODE_EVALUATION_RUNTIME_SLOTS
    )
    eval_misc = tuple(
        _node_eval_slot(node, slot_name)
        for node in nodes
        for slot_name in _NODE_EVALUATION_MISC_SLOTS
    )
    state_representations = tuple(_node_state_representation(node) for node in nodes)
    branch_keys = tuple(_iter_branch_keys(nodes))

    roots: list[ProfileRoot] = [("runner_shell", context.runner)]
    if context.selector is not None:
        roots.append(("selector_linoo_state", context.selector))
    if context.checkpoint_roots:
        roots.append(("checkpoint_state_roots", context.checkpoint_roots))
    roots.extend(
        [
            ("algorithm_node_shells", nodes),
            ("tree_node_shells", tree_nodes),
            ("tree_node_parent_links", parent_links),
            ("tree_node_child_links", child_links),
            ("tree_node_unopened_links", unopened_links),
            ("state_handles", state_handles),
            ("node_eval_shells", node_evaluations),
            ("node_eval_values", eval_values),
            ("node_eval_runtime_states", eval_runtime_states),
            ("node_eval_policy_objective", eval_misc),
            ("branch_keys_ordering_keys", branch_keys),
            ("state_representations", state_representations),
        ]
    )
    if context.evaluator_roots:
        roots.append(("evaluator_model_runtime", context.evaluator_roots))
    if context.runtime is not None:
        roots.append(("remaining_runtime", context.runtime))
    return tuple(roots)


def _iter_branch_keys(nodes: Iterable[object]) -> Iterator[object]:
    for node in nodes:
        yield from _iter_child_branch_refs(_tree_node_slot(node, "branches_children_"))
        yield from _iter_parent_branch_refs(_tree_node_slot(node, "parent_nodes_"))


def _log_exclusive_components(
    *,
    event: str,
    context: RecursiveProfileContext,
    max_objects: int | None,
    max_depth: int | None,
    complete_map: bool,
) -> tuple[int, list[ComponentProfileRecord]]:
    seen: set[int] = set()
    total_bytes = 0
    records: list[ComponentProfileRecord] = []
    shell_components = {
        "runner_shell",
        "algorithm_node_shells",
        "tree_node_shells",
        "node_eval_shells",
    }
    for order, (component, root) in enumerate(_exclusive_components(context), start=1):
        roots = root if isinstance(root, tuple) else (root,)
        if component in shell_components:
            byte_count = _exclusive_shell_size(roots, seen=seen)
            visited_objects = len(seen)
            capped = False
            max_depth_reached_count = 0
            recursion_error_count = 0
        else:
            stats = DeepSizeStats(max_objects=max_objects)
            byte_count = _exclusive_deep_size(
                roots,
                seen=seen,
                max_depth=max_depth,
                stats=stats,
            )
            visited_objects = stats.visited_objects
            capped = stats.capped
            max_depth_reached_count = stats.max_depth_reached_count
            recursion_error_count = stats.recursion_error_count
        total_bytes += byte_count
        records.append(
            ComponentProfileRecord(
                component=f"exclusive:{component}",
                bytes=byte_count,
                visited_objects=visited_objects,
                capped=capped,
                max_depth_reached_count=max_depth_reached_count,
                recursion_error_count=recursion_error_count,
            )
        )
        LOGGER.info(
            "[growth-recursive-profile] event=%s mode=exclusive order=%s "
            "component=%s bytes=%s mb=%s cumulative_mb=%s visited_objects=%s "
            "max_depth=%s complete_map=%s capped=%s max_depth_reached_count=%s "
            "recursion_error_count=%s",
            event,
            order,
            component,
            byte_count,
            format_metric(_mb(byte_count)),
            format_metric(_mb(total_bytes)),
            visited_objects,
            max_depth,
            complete_map,
            capped,
            max_depth_reached_count,
            recursion_error_count,
        )
    return total_bytes, records


def _log_checkpoint_payload_stores(
    *,
    event: str,
    checkpoint_payload_stores: Iterable[CheckpointPayloadStore],
    max_objects: int | None,
    max_depth: int | None,
    complete_map: bool,
) -> list[ComponentProfileRecord]:
    records: list[ComponentProfileRecord] = []
    for index, payload_store in enumerate(checkpoint_payload_stores, start=1):
        stats = DeepSizeStats(max_objects=max_objects)
        byte_count = deep_size(
            payload_store.payloads,
            seen=set(),
            max_depth=max_depth,
            stats=stats,
        )
        records.append(
            ComponentProfileRecord(
                component=f"checkpoint_payload_store:{index}",
                bytes=byte_count,
                visited_objects=stats.visited_objects,
                capped=stats.capped,
                max_depth_reached_count=stats.max_depth_reached_count,
                recursion_error_count=stats.recursion_error_count,
            )
        )
        LOGGER.info(
            "[growth-recursive-profile] event=%s checkpoint_payload_store index=%s "
            "owner_type=%s attr_name=%s mapping_type=%s mapping_length=%s "
            "anchor_count=%s delta_count=%s bytes=%s mb=%s visited_objects=%s "
            "max_depth=%s complete_map=%s capped=%s fully_traversed=%s "
            "max_depth_reached_count=%s "
            "recursion_error_count=%s",
            event,
            index,
            payload_store.owner_type,
            payload_store.attr_name,
            _qualified_type_name(payload_store.payloads),
            len(payload_store.payloads),
            payload_store.anchor_count,
            payload_store.delta_count,
            byte_count,
            format_metric(_mb(byte_count)),
            stats.visited_objects,
            max_depth,
            complete_map,
            stats.capped,
            not stats.capped,
            stats.max_depth_reached_count,
            stats.recursion_error_count,
        )
    return records


def _effective_recursive_max_objects(
    *,
    max_objects: int | None,
    complete_map: bool,
) -> int | None:
    """Return the recursive object cap after applying the uncapped-run guard."""
    if max_objects is not None:
        return max_objects
    if complete_map:
        LOGGER.warning(
            "[growth-recursive-profile] uncapped recursive complete-map mode is "
            "enabled; this may be slow and memory-intensive."
        )
        return None
    LOGGER.warning(
        "[growth-recursive-profile] recursive max_objects=None requested without "
        "complete-map opt-in; using max_objects=%s.",
        _DEFAULT_RECURSIVE_PROFILE_MAX_OBJECTS,
    )
    return _DEFAULT_RECURSIVE_PROFILE_MAX_OBJECTS


def _effective_recursive_max_depth(
    *,
    max_depth: int | None,
    complete_map: bool,
    max_depth_explicit: bool,
) -> int | None:
    """Return the recursive depth cap after applying mode-specific defaults."""
    if max_depth is not None:
        return max_depth
    if max_depth_explicit:
        LOGGER.warning(
            "[growth-recursive-profile] uncapped recursive max_depth=None was "
            "explicitly requested."
        )
        return None
    if complete_map:
        LOGGER.warning(
            "[growth-recursive-profile] recursive complete-map mode defaults to "
            "max_depth=None; this may be slow and memory-intensive."
        )
        return None
    return _DEFAULT_DEEP_SIZE_MAX_DEPTH


def _component_names(records: Iterable[ComponentProfileRecord], *, capped: bool) -> str:
    return (
        "["
        + ",".join(record.component for record in records if record.capped is capped)
        + "]"
    )


def _largest_components(
    records: Iterable[ComponentProfileRecord], *, limit: int
) -> str:
    largest = sorted(records, key=lambda record: record.bytes, reverse=True)[:limit]
    return _format_name_float_pairs(
        (record.component, _mb(record.bytes)) for record in largest
    )


def _log_recursive_profile_summary(
    *,
    event: str,
    rss_mb: float | None,
    total_recursive_reachable_mb: float,
    residual_mb: float | None,
    max_objects: int | None,
    max_depth: int | None,
    complete_map: bool,
    component_records: list[ComponentProfileRecord],
    largest_component_records: list[ComponentProfileRecord],
    checkpoint_histogram: Mapping[str, object],
    checkpoint_payload_store_records: list[ComponentProfileRecord],
) -> None:
    checkpoint_handle_scan_capped = bool(
        checkpoint_histogram.get("handle_scan_capped")
        or checkpoint_histogram.get("handles_scanned_cap_reached")
    )
    checkpoint_payload_store_capped = any(
        record.capped for record in checkpoint_payload_store_records
    )
    LOGGER.info(
        "[growth-recursive-profile-summary] event=%s rss_mb=%s "
        "total_recursive_reachable_mb=%s rss_minus_reachable_mb=%s "
        "capped_components=%s uncapped_components=%s max_objects=%s max_depth=%s "
        "complete_map=%s "
        "checkpoint_handle_scan_capped=%s checkpoint_payload_store_capped=%s "
        "largest_components=%s",
        event,
        format_metric(rss_mb),
        format_metric(total_recursive_reachable_mb),
        format_metric(residual_mb),
        _component_names(component_records, capped=True),
        _component_names(component_records, capped=False),
        max_objects,
        max_depth,
        complete_map,
        checkpoint_handle_scan_capped,
        checkpoint_payload_store_capped,
        _largest_components(largest_component_records, limit=8),
    )


def _log_growth_recursive_profile(
    *,
    runner: object,
    generation: int,
    event: str,
    node_count: int | None,
    branch_count: int | None,
    max_objects: int | None = None,
    max_depth: int | None = None,
    top_n: int = 20,
    complete_map: bool = False,
    max_depth_explicit: bool = False,
    context_node_cap: int | None = None,
) -> None:
    """Log recursive standalone and exclusive memory attribution diagnostics."""
    LOGGER.info(
        "[growth-recursive-profile-enter] event=%s generation=%s "
        "max_objects_arg=%s max_depth_arg=%s max_depth_explicit=%s "
        "complete_map=%s context_node_cap=%s",
        event,
        generation,
        max_objects,
        max_depth,
        max_depth_explicit,
        complete_map,
        context_node_cap,
    )
    effective_max_objects = _effective_recursive_max_objects(
        max_objects=max_objects,
        complete_map=complete_map,
    )
    effective_max_depth = _effective_recursive_max_depth(
        max_depth=max_depth,
        complete_map=complete_map,
        max_depth_explicit=max_depth_explicit,
    )
    context = build_recursive_profile_context(
        runner,
        event=event,
        branch_count=branch_count,
        node_cap=context_node_cap,
    )
    LOGGER.info(
        "[growth-recursive-profile-context-built] event=%s "
        "profile_node_count=%s checkpoint_payload_stores=%s",
        event,
        len(context.nodes),
        len(context.checkpoint_payload_stores),
    )
    rss_mb = current_rss_mb()
    LOGGER.info(
        "[growth-recursive-profile] event=%s generation=%s rss_mb=%s "
        "node_count=%s branch_count=%s profile_node_count=%s max_objects=%s "
        "max_depth=%s complete_map=%s",
        event,
        generation,
        format_metric(rss_mb),
        node_count,
        branch_count,
        len(context.nodes),
        effective_max_objects,
        effective_max_depth,
        complete_map,
    )

    standalone_records = _log_standalone_components(
        event=event,
        context=context,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
        complete_map=complete_map,
    )
    exclusive_total_bytes, exclusive_records = _log_exclusive_components(
        event=event,
        context=context,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
        complete_map=complete_map,
    )

    _log_histogram(event, "tree_topology", tree_topology_histograms(context.nodes))
    _log_histogram(
        event,
        "tree_parent_links",
        parent_link_storage_histogram(
            context.nodes,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "tree_node_child_links_detail",
        child_link_storage_detail_histogram(
            context.nodes,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "state_handle_materialization_detail",
        state_handle_materialization_detail_histogram(
            context.nodes,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "state_retention_by_node_status",
        state_retention_by_node_status_histogram(
            context.nodes,
            selector=context.selector,
        ),
    )
    _log_histogram(
        event,
        "state_eviction_runtime",
        state_eviction_runtime_histogram(context.runner),
    )
    _log_histogram(
        event,
        "node_evaluation_runtime",
        node_evaluation_runtime_histograms(context.nodes),
    )
    for payload in node_evaluation_runtime_detail_histograms(
        context.nodes,
        max_depth=effective_max_depth,
        max_objects=effective_max_objects,
    ):
        _log_histogram(event, "node_evaluation_runtime_detail", payload)
    _log_histogram(
        event,
        "linoo",
        linoo_state_histograms(context.selector, max_depth=effective_max_depth),
    )
    for payload in linoo_deep_breakdown_histograms(
        context.selector,
        max_depth=effective_max_depth,
        max_objects=effective_max_objects,
    ):
        _log_histogram(event, "linoo_deep_breakdown", payload)
    _log_histogram(
        event,
        "linoo_node_state_table",
        linoo_node_state_table_histogram(
            context.selector,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "linoo_node_state_slots",
        linoo_node_state_slots_histogram(
            context.selector,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "linoo_candidate_heaps",
        linoo_candidate_heap_histogram(
            context.selector,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_histogram(
        event,
        "linoo_selector_detail",
        linoo_selector_detail_histogram(
            context.selector,
            max_depth=effective_max_depth,
            max_objects=effective_max_objects,
        ),
    )
    _log_gc_shallow_size_summary(event=event, top_n=top_n)
    checkpoint_histogram = checkpoint_state_histograms(
        context.nodes,
        context.checkpoint_payload_stores,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
    )
    _log_histogram(
        event,
        "checkpoint_state",
        checkpoint_histogram,
    )
    checkpoint_payload_store_records = _log_checkpoint_payload_stores(
        event=event,
        checkpoint_payload_stores=context.checkpoint_payload_stores,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
        complete_map=complete_map,
    )
    for payload in checkpoint_payload_lifetime_histograms(
        context.nodes,
        context.checkpoint_payload_stores,
        checkpoint_payload_store_records=checkpoint_payload_store_records,
    ):
        _log_histogram(
            event,
            "checkpoint_payload_lifetime",
            payload,
        )
    for payload in checkpoint_payload_shape_histograms(
        context.checkpoint_payload_stores,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
    ):
        _log_histogram(
            event,
            "checkpoint_payload_shape",
            payload,
        )
    _log_histogram(
        event,
        "checkpoint_state_roots_detail",
        checkpoint_state_roots_detail_histogram(
            context.checkpoint_payload_stores,
            max_objects=effective_max_objects,
            max_depth=effective_max_depth,
        ),
    )

    total_recursive_reachable_mb = _mb(exclusive_total_bytes)
    residual_mb = None if rss_mb is None else rss_mb - total_recursive_reachable_mb
    LOGGER.info(
        "[growth-recursive-profile] event=%s total_recursive_reachable_mb=%s "
        "rss_mb=%s residual rss_minus_reachable_mb=%s",
        event,
        format_metric(total_recursive_reachable_mb),
        format_metric(rss_mb),
        format_metric(residual_mb),
    )
    _log_recursive_profile_summary(
        event=event,
        rss_mb=rss_mb,
        total_recursive_reachable_mb=total_recursive_reachable_mb,
        residual_mb=residual_mb,
        max_objects=effective_max_objects,
        max_depth=effective_max_depth,
        complete_map=complete_map,
        component_records=[
            *standalone_records,
            *exclusive_records,
            *checkpoint_payload_store_records,
        ],
        largest_component_records=exclusive_records,
        checkpoint_histogram=checkpoint_histogram,
        checkpoint_payload_store_records=checkpoint_payload_store_records,
    )


log_growth_recursive_memory_profile = _log_growth_recursive_profile

__all__ = ["log_growth_recursive_memory_profile"]
