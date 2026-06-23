"""Recursive restored-tree memory diagnostics for Morpion growth runtimes."""

from __future__ import annotations

import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    FunctionType,
    MethodType,
    ModuleType,
)
from typing import cast

from .pipeline_memory import current_rss_mb, format_metric

LOGGER = logging.getLogger(__name__)

_ATOMIC_TYPES = (str, bytes, bytearray, int, float, bool, type(None))
_SKIP_DEEP_TYPES = (
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    CodeType,
    type,
)
_CONTAINER_TYPES = (list, tuple, set, frozenset)
_CONTAINER_VALUE_TYPES = (Mapping, list, tuple, set, frozenset)

_RUNTIME_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("profile_search_root",),
    ("profile_runtime_root",),
    ("_runtime",),
    ("runtime",),
    ("search",),
    ("_search",),
)
_SELECTOR_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("profile_selector",),
    ("_runtime", "node_selector"),
    ("_runtime", "selector"),
    ("runtime", "node_selector"),
    ("runtime", "selector"),
    ("node_selector",),
    ("selector",),
)
_CHECKPOINT_ROOT_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("profile_state_resolver",),
    ("profile_checkpoint_state_resolver",),
    ("profile_state_codec",),
    ("_state_codec",),
    ("state_codec",),
    ("_checkpoint_state_resolver",),
    ("checkpoint_state_resolver",),
)
_EVALUATOR_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("profile_evaluator_bundle",),
    ("_runtime", "master_state_evaluator"),
    ("_runtime", "state_evaluator"),
    ("_runtime", "evaluator"),
    ("runtime", "master_state_evaluator"),
    ("runtime", "state_evaluator"),
    ("runtime", "evaluator"),
    ("_current_evaluator_bundle",),
    ("current_evaluator_bundle",),
)

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

_ANCHOR_PAYLOAD_TYPE_SUFFIX = "AnchorCheckpointStatePayload"
_DELTA_PAYLOAD_TYPE_SUFFIX = "DeltaCheckpointStatePayload"

type ProfileRoot = tuple[str, object]


@dataclass(slots=True)
class DeepSizeStats:
    """Mutable counters for one recursive-size traversal."""

    visited_objects: int = 0
    max_objects: int | None = None
    capped: bool = False


@dataclass(frozen=True, slots=True)
class RecursiveProfileContext:
    """Resolved roots used by one recursive growth-memory profile pass."""

    runner: object
    runtime: object | None
    selector: object | None
    checkpoint_roots: tuple[object, ...]
    evaluator_roots: tuple[object, ...]
    nodes: tuple[object, ...]


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    module = value_type.__module__
    qualname = value_type.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def _mb(byte_count: int) -> float:
    return byte_count / (1024 * 1024)


def _size_or_zero(value: object) -> int:
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def _should_skip_deep(value: object) -> bool:
    return isinstance(value, _SKIP_DEEP_TYPES)


def _raw_getattr(value: object, attr_name: str) -> object | None:
    """Read a concrete attribute/slot without using ``dir`` or properties."""
    try:
        result: object = object.__getattribute__(value, attr_name)
    except Exception:
        return None
    return result


def _safe_call_no_args(value: object) -> object | None:
    if not callable(value):
        return value
    try:
        return cast("Callable[[], object]", value)()
    except Exception:
        return None


def _raw_attr_path(root: object, path: tuple[str, ...]) -> object | None:
    value: object | None = root
    for index, attr_name in enumerate(path):
        if value is None:
            return None
        value = _raw_getattr(value, attr_name)
        if (
            index == len(path) - 1
            and attr_name.startswith("profile_")
            and callable(value)
        ):
            value = _safe_call_no_args(value)
    return value


def _first_attr_path(root: object, paths: tuple[tuple[str, ...], ...]) -> object | None:
    for path in paths:
        value = _raw_attr_path(root, path)
        if value is not None:
            return value
    return None


def _all_attr_paths(
    root: object, paths: tuple[tuple[str, ...], ...]
) -> tuple[object, ...]:
    values: list[object] = []
    seen_ids: set[int] = set()
    for path in paths:
        value = _raw_attr_path(root, path)
        if value is None:
            continue
        value_id = id(value)
        if value_id in seen_ids:
            continue
        seen_ids.add(value_id)
        values.append(value)
    return tuple(values)


def _safe_object_dict(value: object) -> Mapping[object, object] | None:
    raw_dict = _raw_getattr(value, "__dict__")
    if isinstance(raw_dict, Mapping):
        return raw_dict
    return None


def slot_names(type_or_obj: object) -> tuple[str, ...]:
    """Return declared slot names across the MRO without consulting ``dir``."""
    value_type = type_or_obj if isinstance(type_or_obj, type) else type(type_or_obj)
    seen: set[str] = set()
    ordered_names: list[str] = []
    for base_type in value_type.__mro__:
        raw_slots = getattr(base_type, "__slots__", ())
        candidate_names: tuple[object, ...]
        if isinstance(raw_slots, str):
            candidate_names = (raw_slots,)
        else:
            try:
                candidate_names = tuple(raw_slots)
            except TypeError:
                candidate_names = ()
        for slot_name in candidate_names:
            if not isinstance(slot_name, str):
                continue
            if slot_name in {"__weakref__", "__dict__"} or slot_name in seen:
                continue
            seen.add(slot_name)
            ordered_names.append(slot_name)
    return tuple(ordered_names)


def _iter_object_attribute_values(value: object) -> Iterator[object]:
    raw_dict = _safe_object_dict(value)
    if raw_dict is not None:
        yield raw_dict
    for slot_name in slot_names(value):
        slot_value = _raw_getattr(value, slot_name)
        if slot_value is not None:
            yield slot_value


def deep_size(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None = None,
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


def _deep_size(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None,
    depth: int,
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
    size = _size_or_zero(obj)

    if isinstance(obj, _ATOMIC_TYPES) or _should_skip_deep(obj):
        return size
    if max_depth is not None and depth >= max_depth:
        return size

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            size += _deep_size(
                key,
                seen=seen,
                max_depth=max_depth,
                depth=depth + 1,
                stats=stats,
            )
            size += _deep_size(
                value,
                seen=seen,
                max_depth=max_depth,
                depth=depth + 1,
                stats=stats,
            )
        return size

    if isinstance(obj, _CONTAINER_TYPES):
        for item in obj:
            size += _deep_size(
                item,
                seen=seen,
                max_depth=max_depth,
                depth=depth + 1,
                stats=stats,
            )
        return size

    for attr_value in _iter_object_attribute_values(obj):
        size += _deep_size(
            attr_value,
            seen=seen,
            max_depth=max_depth,
            depth=depth + 1,
            stats=stats,
        )
    return size


def _iter_from_candidate(candidate: object) -> Iterator[object] | None:
    if isinstance(candidate, Mapping):
        return iter(candidate.values())
    if isinstance(candidate, str | bytes | bytearray):
        return None
    try:
        return iter(candidate) if isinstance(candidate, Iterable) else None
    except TypeError:
        return None


def _profile_nodes_from_runner(runner: object) -> tuple[object, ...]:
    for method_name in (
        "profile_iter_nodes",
        "iter_profile_nodes",
        "_profile_iter_nodes",
        "iter_nodes",
        "live_nodes",
        "nodes",
    ):
        method = _raw_getattr(runner, method_name)
        if not callable(method):
            continue
        try:
            iterator = _iter_from_candidate(cast("Callable[[], object]", method)())
        except Exception:
            continue
        if iterator is not None:
            return tuple(iterator)

    for attr_path in (
        ("node_store",),
        ("nodes",),
        ("tree", "nodes"),
        ("search_tree", "nodes"),
        ("runtime", "nodes"),
        ("runtime", "tree", "nodes"),
        ("runtime", "search_tree", "nodes"),
        ("_runtime", "nodes"),
        ("_runtime", "tree", "nodes"),
        ("_runtime", "search_tree", "nodes"),
        ("_runtime", "node_store"),
        ("_runtime", "node_store", "nodes"),
        ("_runtime", "_nodes"),
    ):
        value = _raw_attr_path(runner, attr_path)
        if value is None:
            continue
        iterator = _iter_from_candidate(value)
        if iterator is not None:
            return tuple(iterator)
    return ()


def build_recursive_profile_context(runner: object) -> RecursiveProfileContext:
    """Resolve profile roots once, without forcing lazy runtime properties."""
    return RecursiveProfileContext(
        runner=runner,
        runtime=_first_attr_path(runner, _RUNTIME_ATTR_PATHS),
        selector=_first_attr_path(runner, _SELECTOR_ATTR_PATHS),
        checkpoint_roots=_all_attr_paths(runner, _CHECKPOINT_ROOT_ATTR_PATHS),
        evaluator_roots=_all_attr_paths(runner, _EVALUATOR_ATTR_PATHS),
        nodes=_profile_nodes_from_runner(runner),
    )


def _node_tree_node(node: object) -> object | None:
    return _raw_getattr(node, "tree_node")


def _node_tree_evaluation(node: object) -> object | None:
    return _raw_getattr(node, "tree_evaluation") or _raw_getattr(
        node,
        "node_evaluation",
    )


def _node_state_representation(node: object) -> object | None:
    return _raw_getattr(node, "_state_representation")


def _tree_node_slot(node: object, slot_name: str) -> object | None:
    tree_node = _node_tree_node(node)
    if tree_node is None:
        return None
    return _raw_getattr(tree_node, slot_name)


def _node_eval_slot(node: object, slot_name: str) -> object | None:
    node_eval = _node_tree_evaluation(node)
    if node_eval is None:
        return None
    return _raw_getattr(node_eval, slot_name)


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
    stats: DeepSizeStats,
) -> int:
    total = 0
    for root in roots:
        if root is None:
            continue
        total += deep_size(root, seen=seen, stats=stats)
    return total


def _exclusive_shell_size(roots: Iterable[object | None], *, seen: set[int]) -> int:
    total = 0
    for root in roots:
        if root is None:
            continue
        total += _shallow_object_and_dict_size(root, seen=seen)
    return total


def _iter_parent_branch_refs(parent_nodes: object | None) -> Iterator[object]:
    if not isinstance(parent_nodes, Mapping):
        return
    for branch_set in parent_nodes.values():
        if isinstance(branch_set, Iterable) and not isinstance(
            branch_set,
            str | bytes | bytearray,
        ):
            yield from branch_set


def _iter_child_branch_refs(branches_children: object | None) -> Iterator[object]:
    if isinstance(branches_children, Mapping):
        yield from branches_children.keys()
        return
    branch = _raw_getattr(branches_children, "branch") if branches_children else None
    if branch is not None:
        yield branch


def _component_roots(context: RecursiveProfileContext) -> tuple[ProfileRoot, ...]:
    node_tree_nodes = tuple(
        tree_node
        for node in context.nodes
        if (tree_node := _node_tree_node(node)) is not None
    )
    node_evaluations = tuple(
        node_eval
        for node in context.nodes
        if (node_eval := _node_tree_evaluation(node)) is not None
    )
    state_handles = tuple(
        handle
        for node in context.nodes
        if (handle := _tree_node_slot(node, "state_handle_")) is not None
    )
    roots: list[ProfileRoot] = [
        ("all_profile_nodes", context.nodes),
        ("tree_node_structures", node_tree_nodes),
        ("node_evaluations", node_evaluations),
        ("state_handles", state_handles),
    ]
    if context.runtime is not None:
        roots.append(("runtime_root", context.runtime))
    if context.selector is not None:
        roots.append(("linoo_selector", context.selector))
    if context.checkpoint_roots:
        roots.append(("checkpoint_state_roots", context.checkpoint_roots))
    if context.evaluator_roots:
        roots.append(("evaluator_model_runtime", context.evaluator_roots))
    return tuple(roots)


def tree_topology_histograms(nodes: Iterable[object]) -> dict[str, object]:
    """Return topology histograms without materializing child/parent properties."""
    child_link_count_hist = Counter[str]()
    parent_count_hist = Counter[str]()
    total_parent_branch_refs_hist = Counter[str]()
    branches_children_type_hist = Counter[str]()
    parent_nodes_type_hist = Counter[str]()
    non_opened_branches_type_hist = Counter[str]()

    for node in nodes:
        tree_node = _node_tree_node(node) or node
        branches_children = _raw_getattr(tree_node, "branches_children_")
        parent_nodes = _raw_getattr(tree_node, "parent_nodes_")
        non_opened_branches = _raw_getattr(tree_node, "non_opened_branches_")

        branches_children_type_hist[_qualified_type_name(branches_children)] += 1
        parent_nodes_type_hist[_qualified_type_name(parent_nodes)] += 1
        non_opened_branches_type_hist[_qualified_type_name(non_opened_branches)] += 1

        child_count = _child_link_count_from_storage(branches_children)
        child_link_count_hist[str(child_count)] += 1

        parent_count = len(parent_nodes) if isinstance(parent_nodes, Mapping) else 0
        parent_count_hist[str(parent_count)] += 1
        total_parent_refs = 0
        if isinstance(parent_nodes, Mapping):
            for branch_set in parent_nodes.values():
                if isinstance(branch_set, Iterable) and not isinstance(
                    branch_set,
                    str | bytes | bytearray,
                ):
                    total_parent_refs += sum(1 for _item in branch_set)
                else:
                    total_parent_refs += 1
        total_parent_branch_refs_hist[str(total_parent_refs)] += 1

    return {
        "child_link_count": dict(child_link_count_hist),
        "parent_nodes_len": dict(parent_count_hist),
        "parent_branch_refs": dict(total_parent_branch_refs_hist),
        "branches_children_types": dict(branches_children_type_hist),
        "parent_nodes_types": dict(parent_nodes_type_hist),
        "non_opened_branches_types": dict(non_opened_branches_type_hist),
    }


def _child_link_count_from_storage(branches_children: object | None) -> int:
    if branches_children is None:
        return 0
    if isinstance(branches_children, Mapping):
        return len(branches_children)
    if _raw_getattr(branches_children, "branch") is not None:
        return 1
    if isinstance(branches_children, tuple) and len(branches_children) == 2:
        return 1
    if isinstance(branches_children, Sized):
        return len(branches_children)
    return 1


def node_evaluation_runtime_histograms(nodes: Iterable[object]) -> dict[str, object]:
    """Return materialized NodeMaxEvaluation runtime-state counters."""
    counts = Counter[str]()
    eval_type_counts = Counter[str]()
    for node in nodes:
        node_eval = _node_tree_evaluation(node)
        if node_eval is None:
            continue
        eval_type_counts[_qualified_type_name(node_eval)] += 1
        for slot_name in _NODE_EVALUATION_RUNTIME_SLOTS:
            if _raw_getattr(node_eval, slot_name) is not None:
                counts[f"{slot_name}_non_none"] += 1
        for slot_name in _NODE_EVALUATION_VALUE_SLOTS:
            if _raw_getattr(node_eval, slot_name) is not None:
                counts[f"{slot_name}_non_none"] += 1
    return {
        "node_evaluation_types": dict(eval_type_counts),
        "runtime_state_counts": dict(counts),
    }


def linoo_state_histograms(selector: object | None) -> dict[str, object]:
    """Return sparse Linoo state-table diagnostics when a Linoo selector is present."""
    if selector is None:
        return {"present": False}
    node_state_by_id = _raw_getattr(selector, "_node_state_by_id")
    if not isinstance(node_state_by_id, Mapping):
        return {
            "present": True,
            "node_state_table_type": _qualified_type_name(node_state_by_id),
        }

    default_count = 0
    non_default_count = 0
    state_type_counts = Counter[str]()
    slot_value_type_counts = Counter[str]()
    container_shallow_total = 0
    container_recursive_seen: set[int] = set()
    container_recursive_total = 0

    for state in node_state_by_id.values():
        state_type_counts[_qualified_type_name(state)] += 1
        is_default = _raw_getattr(state, "is_default")
        if callable(is_default):
            try:
                if bool(cast("Callable[[], object]", is_default)()):
                    default_count += 1
                else:
                    non_default_count += 1
            except Exception:
                non_default_count += 1
        for slot_name in slot_names(state):
            slot_value = _raw_getattr(state, slot_name)
            if slot_value is None:
                continue
            slot_value_type_counts[_qualified_type_name(slot_value)] += 1
            if isinstance(slot_value, _CONTAINER_VALUE_TYPES):
                container_shallow_total += _size_or_zero(slot_value)
                container_recursive_total += deep_size(
                    slot_value,
                    seen=container_recursive_seen,
                )

    return {
        "present": True,
        "node_state_count": len(node_state_by_id),
        "default_count": default_count,
        "non_default_count": non_default_count,
        "node_state_table_type": _qualified_type_name(node_state_by_id),
        "state_types": dict(state_type_counts),
        "slot_value_types": dict(slot_value_type_counts),
        "container_shallow_bytes": container_shallow_total,
        "container_recursive_bytes": container_recursive_total,
    }


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
        value = _raw_getattr(handle, attr_name)
        if value is not None:
            return value
    return None


def _resolver_payloads(resolver: object) -> Mapping[object, object] | None:
    """Return checkpoint payload storage from known resolver layouts."""
    for attr_name in (
        "state_payloads_by_node_id",
        "_state_payloads_by_node_id",
        "payloads_by_node_id",
        "_payloads_by_node_id",
    ):
        value = _raw_getattr(resolver, attr_name)
        if isinstance(value, Mapping):
            return value
    return None


def checkpoint_state_histograms(nodes: Iterable[object]) -> dict[str, object]:
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
    resolved_recursive_seen: set[int] = set()
    resolved_recursive_bytes = 0

    for node in nodes:
        handle = _tree_node_slot(node, "state_handle_")
        if handle is None:
            continue
        handle_type_counts[_qualified_type_name(handle)] += 1

        state_value = _raw_getattr(handle, "state_")
        if state_value is not None:
            materialized_state_count += 1
            resolved_state_ids.add(id(state_value))
            resolved_recursive_bytes += deep_size(
                state_value, seen=resolved_recursive_seen
            )

        resolver = _handle_resolver(handle)
        if resolver is None:
            continue
        resolver_ids.add(id(resolver))
        node_id = _raw_getattr(handle, "node_id")
        payloads = _resolver_payloads(resolver)
        if isinstance(node_id, int) and isinstance(payloads, Mapping):
            payload = payloads.get(node_id)
            if payload is not None and id(payload) not in payload_ids:
                payload_ids.add(id(payload))
                payload_type = _qualified_type_name(payload)
                if payload_type.endswith(_ANCHOR_PAYLOAD_TYPE_SUFFIX):
                    anchor_count += 1
                elif payload_type.endswith(_DELTA_PAYLOAD_TYPE_SUFFIX):
                    delta_count += 1
                payload_recursive_bytes += deep_size(
                    payload, seen=payload_recursive_seen
                )
        resolved_states = _raw_getattr(resolver, "_resolved_states")
        if isinstance(resolved_states, Mapping):
            for state in resolved_states.values():
                if id(state) in resolved_state_ids:
                    continue
                resolved_state_ids.add(id(state))
                materialized_state_count += 1
                resolved_recursive_bytes += deep_size(
                    state, seen=resolved_recursive_seen
                )

    return {
        "handle_types": dict(handle_type_counts),
        "resolver_count": len(resolver_ids),
        "anchor_payload_count": anchor_count,
        "delta_payload_count": delta_count,
        "payload_recursive_bytes": payload_recursive_bytes,
        "materialized_state_count": materialized_state_count,
        "materialized_state_recursive_bytes": resolved_recursive_bytes,
    }


def _log_histogram(event: str, name: str, payload: Mapping[str, object]) -> None:
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


def _log_standalone_components(
    *,
    event: str,
    context: RecursiveProfileContext,
    max_objects: int | None,
) -> None:
    for component, root in _component_roots(context):
        stats = DeepSizeStats(max_objects=max_objects)
        byte_count = deep_size(root, seen=set(), stats=stats)
        LOGGER.info(
            "[growth-recursive-profile] event=%s mode=standalone component=%s "
            "bytes=%s mb=%s visited_objects=%s capped=%s",
            event,
            component,
            byte_count,
            format_metric(_mb(byte_count)),
            stats.visited_objects,
            stats.capped,
        )


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
) -> int:
    seen: set[int] = set()
    total_bytes = 0
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
        else:
            stats = DeepSizeStats(max_objects=max_objects)
            byte_count = _exclusive_deep_size(roots, seen=seen, stats=stats)
            visited_objects = stats.visited_objects
            capped = stats.capped
        total_bytes += byte_count
        LOGGER.info(
            "[growth-recursive-profile] event=%s mode=exclusive order=%s "
            "component=%s bytes=%s mb=%s cumulative_mb=%s visited_objects=%s "
            "capped=%s",
            event,
            order,
            component,
            byte_count,
            format_metric(_mb(byte_count)),
            format_metric(_mb(total_bytes)),
            visited_objects,
            capped,
        )
    return total_bytes


def log_growth_recursive_memory_profile(
    *,
    runner: object,
    generation: int,
    event: str,
    node_count: int | None,
    branch_count: int | None,
    max_objects: int | None = None,
) -> None:
    """Log recursive standalone and exclusive memory attribution diagnostics."""
    context = build_recursive_profile_context(runner)
    rss_mb = current_rss_mb()
    LOGGER.info(
        "[growth-recursive-profile] event=%s generation=%s rss_mb=%s "
        "node_count=%s branch_count=%s profile_node_count=%s max_objects=%s",
        event,
        generation,
        format_metric(rss_mb),
        node_count,
        branch_count,
        len(context.nodes),
        max_objects,
    )

    _log_standalone_components(
        event=event,
        context=context,
        max_objects=max_objects,
    )
    exclusive_total_bytes = _log_exclusive_components(
        event=event,
        context=context,
        max_objects=max_objects,
    )

    _log_histogram(event, "tree_topology", tree_topology_histograms(context.nodes))
    _log_histogram(
        event,
        "node_evaluation_runtime",
        node_evaluation_runtime_histograms(context.nodes),
    )
    _log_histogram(event, "linoo", linoo_state_histograms(context.selector))
    _log_histogram(
        event, "checkpoint_state", checkpoint_state_histograms(context.nodes)
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


__all__ = [
    "DeepSizeStats",
    "build_recursive_profile_context",
    "checkpoint_state_histograms",
    "deep_size",
    "linoo_state_histograms",
    "log_growth_recursive_memory_profile",
    "node_evaluation_runtime_histograms",
    "slot_names",
    "tree_topology_histograms",
]
