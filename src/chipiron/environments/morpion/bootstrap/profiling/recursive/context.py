"""Context discovery for Morpion recursive memory profiling."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.bootstrap.pipeline_memory import format_metric

from .object_access import (
    ATOMIC_TYPES,
    CONTAINER_TYPES,
    all_attr_paths,
    first_attr_path,
    iter_object_attribute_values,
    qualified_type_name,
    raw_attr_path,
    raw_getattr,
    safe_object_dict,
    should_skip_deep,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)

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
    ("_live_compact_state_resolver",),
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
_DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP = 100
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
_LINOO_NODE_STATE_TABLE_ATTR_NAME = "_node_state_by_id"
_KNOWN_CHECKPOINT_CANDIDATE_PATHS: tuple[tuple[str, ...], ...] = tuple(
    dict.fromkeys((
        *_CHECKPOINT_ROOT_ATTR_PATHS,
        ("checkpoint_state_resolver",),
        ("_checkpoint_state_resolver",),
        ("_runtime", "checkpoint_state_resolver"),
        ("_runtime", "_checkpoint_state_resolver"),
        ("runtime", "checkpoint_state_resolver"),
        ("runtime", "_checkpoint_state_resolver"),
    ))
)

type ProfileRoot = tuple[str, object]

__all__ = [
    "CheckpointPayloadStore",
    "ComponentProfileRecord",
    "RecursiveProfileContext",
    "build_recursive_profile_context",
    "component_roots",
    "unique_roots",
]


@dataclass(frozen=True, slots=True)
class CheckpointPayloadStore:
    """Concrete mapping that owns checkpoint state payload objects."""

    resolver_type: str
    resolver_id: int
    owner_type: str
    attr_name: str
    payloads: Mapping[object, object]
    anchor_count: int
    delta_count: int


@dataclass(frozen=True, slots=True)
class ComponentProfileRecord:
    """One logged recursive component measurement."""

    component: str
    bytes: int
    visited_objects: int
    capped: bool
    max_depth_reached_count: int = 0
    recursion_error_count: int = 0


@dataclass(frozen=True, slots=True)
class RecursiveProfileContext:
    """Resolved roots used by one recursive growth-memory profile pass."""

    runner: object
    runtime: object | None
    selector: object | None
    checkpoint_roots: tuple[object, ...]
    checkpoint_payload_stores: tuple[CheckpointPayloadStore, ...]
    evaluator_roots: tuple[object, ...]
    nodes: tuple[object, ...]


def build_recursive_profile_context(
    runner: object,
    *,
    event: str = "unknown",
    branch_count: int | None = None,
    node_cap: int | None = None,
) -> RecursiveProfileContext:
    """Resolve profile roots once, without forcing lazy runtime properties."""
    start_time = time.perf_counter()
    LOGGER.info("[growth-recursive-profile] event=%s context_build_start", event)
    runtime = first_attr_path(runner, _RUNTIME_ATTR_PATHS)
    selector_root = first_attr_path(runner, _SELECTOR_ATTR_PATHS)
    linoo_selector = find_linoo_selector_root(selector_root)
    if linoo_selector is None and runtime is not None:
        linoo_selector = find_linoo_selector_root(runtime)
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_runtime_found "
        "runtime_found=%s selector_found=%s",
        event,
        runtime is not None,
        (linoo_selector or selector_root) is not None,
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_nodes_start node_cap=%s",
        event,
        node_cap,
    )
    nodes_start = time.perf_counter()
    nodes = profile_nodes_from_runner_capped(runner, node_cap=node_cap)
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_nodes_done "
        "node_count=%s elapsed_s=%s",
        event,
        len(nodes),
        format_metric(time.perf_counter() - nodes_start),
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_branches_start",
        event,
    )
    branches_start = time.perf_counter()
    resolved_branch_count = (
        branch_count if branch_count is not None else count_profile_branches(nodes)
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_branches_done "
        "branch_count=%s elapsed_s=%s",
        event,
        resolved_branch_count,
        format_metric(time.perf_counter() - branches_start),
    )
    checkpoint_roots = all_attr_paths(runner, _CHECKPOINT_ROOT_ATTR_PATHS)
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_checkpoint_stores_start",
        event,
    )
    checkpoint_stores_start = time.perf_counter()
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_checkpoint_stores_known_paths_start",
        event,
    )
    known_paths_checkpoint_stores_start = time.perf_counter()
    checkpoint_payload_stores = find_checkpoint_payload_stores_from_known_paths_only(
        runner=runner,
        runtime=runtime,
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_checkpoint_stores_known_paths_done "
        "count=%s elapsed_s=%s",
        event,
        len(checkpoint_payload_stores),
        format_metric(time.perf_counter() - known_paths_checkpoint_stores_start),
    )
    if not checkpoint_payload_stores:
        handle_cap = checkpoint_store_handle_discovery_cap(node_cap=node_cap)
        LOGGER.info(
            "[growth-recursive-profile] event=%s "
            "context_build_checkpoint_stores_handle_fallback_start handle_cap=%s",
            event,
            handle_cap,
        )
        fallback_checkpoint_stores_start = time.perf_counter()
        checkpoint_payload_stores = find_checkpoint_payload_stores_from_handle_fallback(
            nodes,
            handle_cap=handle_cap,
        )
        LOGGER.info(
            "[growth-recursive-profile] event=%s "
            "context_build_checkpoint_stores_handle_fallback_done count=%s elapsed_s=%s",
            event,
            len(checkpoint_payload_stores),
            format_metric(time.perf_counter() - fallback_checkpoint_stores_start),
        )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_checkpoint_stores_done "
        "count=%s elapsed_s=%s",
        event,
        len(checkpoint_payload_stores),
        format_metric(time.perf_counter() - checkpoint_stores_start),
    )
    checkpoint_roots_with_payloads = unique_roots((
        *checkpoint_roots,
        *(store.payloads for store in checkpoint_payload_stores),
    ))
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_done total_elapsed_s=%s",
        event,
        format_metric(time.perf_counter() - start_time),
    )

    return RecursiveProfileContext(
        runner=runner,
        runtime=runtime,
        selector=linoo_selector or selector_root,
        checkpoint_roots=checkpoint_roots_with_payloads,
        checkpoint_payload_stores=checkpoint_payload_stores,
        evaluator_roots=all_attr_paths(runner, _EVALUATOR_ATTR_PATHS),
        nodes=nodes,
    )


def unique_roots(roots: Iterable[object | None]) -> tuple[object, ...]:
    """Return non-None roots deduplicated by identity."""
    unique: list[object] = []
    seen_ids: set[int] = set()
    for root in roots:
        if root is None:
            continue
        root_id = id(root)
        if root_id in seen_ids:
            continue
        seen_ids.add(root_id)
        unique.append(root)
    return tuple(unique)


def component_roots(context: RecursiveProfileContext) -> tuple[ProfileRoot, ...]:
    """Return top-level roots for standalone recursive component sizing."""
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


def profile_nodes_from_runner(runner: object) -> tuple[object, ...]:
    """Return all profile nodes discoverable from a runner."""
    return profile_nodes_from_runner_capped(runner, node_cap=None)


def profile_nodes_from_runner_capped(
    runner: object,
    *,
    node_cap: int | None,
) -> tuple[object, ...]:
    """Return runner profile nodes, stopping at ``node_cap`` when provided."""
    for method_name in (
        "profile_iter_nodes",
        "iter_profile_nodes",
        "_profile_iter_nodes",
        "iter_nodes",
        "live_nodes",
        "nodes",
    ):
        method = raw_getattr(runner, method_name)
        if not callable(method):
            continue
        try:
            iterator = _iter_from_candidate(cast("Callable[[], object]", method)())
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if iterator is not None:
            return materialize_profile_nodes(iterator, node_cap=node_cap)

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
        value = raw_attr_path(runner, attr_path)
        if value is None:
            continue
        iterator = _iter_from_candidate(value)
        if iterator is not None:
            return materialize_profile_nodes(iterator, node_cap=node_cap)
    return ()


def materialize_profile_nodes(
    values: Iterable[object],
    *,
    node_cap: int | None,
) -> tuple[object, ...]:
    """Materialize profile nodes with an optional count cap."""
    if node_cap is None:
        return tuple(values)
    materialized: list[object] = []
    for value in values:
        materialized.append(value)
        if len(materialized) >= node_cap:
            break
    return tuple(materialized)


def count_profile_branches(nodes: Iterable[object]) -> int:
    """Count child and parent branch references across profile nodes."""
    branch_count = 0
    for node in nodes:
        branch_count += sum(
            1
            for _ in _iter_child_branch_refs(
                _tree_node_slot(node, "branches_children_")
            )
        )
        branch_count += sum(
            1 for _ in _iter_parent_branch_refs(_tree_node_slot(node, "parent_nodes_"))
        )
    return branch_count


def checkpoint_store_handle_discovery_cap(
    *,
    node_cap: int | None,
    handle_cap: int = _DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP,
) -> int:
    """Return the bounded handle-scan cap used for checkpoint store discovery."""
    if node_cap is None:
        return handle_cap
    return min(node_cap, handle_cap)


def find_linoo_selector_root(root: object | None) -> object | None:
    """Find the concrete nested Linoo selector without materializing properties."""
    if root is None:
        return None

    seen: set[int] = set()
    stack: list[object] = [root]
    while stack:
        value = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)

        if isinstance(value, ATOMIC_TYPES) or should_skip_deep(value):
            continue

        node_state_by_id = raw_getattr(value, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
        if isinstance(node_state_by_id, Mapping):
            return value

        stack.extend(_iter_linoo_selector_search_children(value))
    return None


def find_checkpoint_payload_stores_from_known_paths_only(
    *,
    runner: object,
    runtime: object | None,
) -> tuple[CheckpointPayloadStore, ...]:
    """Find checkpoint payload stores from known resolver paths."""
    stores: list[CheckpointPayloadStore] = []
    checked_mapping_ids: set[int] = set()
    payload_mapping_ids: set[int] = set()
    seen_candidate_ids: set[int] = set()

    def add_candidate(candidate: object | None) -> None:
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen_candidate_ids:
            return
        seen_candidate_ids.add(candidate_id)
        _append_checkpoint_payload_stores_from_shallow_candidate(
            candidate,
            stores=stores,
            checked_mapping_ids=checked_mapping_ids,
            payload_mapping_ids=payload_mapping_ids,
        )

    for attr_path in _KNOWN_CHECKPOINT_CANDIDATE_PATHS:
        add_candidate(raw_attr_path(runner, attr_path))
    if runtime is not None:
        add_candidate(runtime)
        for attr_path in _CHECKPOINT_ROOT_ATTR_PATHS:
            add_candidate(raw_attr_path(runtime, attr_path))
    return tuple(stores)


def find_checkpoint_payload_stores_from_handle_fallback(
    nodes: Sequence[object],
    *,
    handle_cap: int,
) -> tuple[CheckpointPayloadStore, ...]:
    """Find checkpoint payload stores from profile-node state handles."""
    stores: list[CheckpointPayloadStore] = []
    checked_mapping_ids: set[int] = set()
    payload_mapping_ids: set[int] = set()
    for node in nodes[:handle_cap]:
        handle = _tree_node_slot(node, "state_handle_")
        if handle is None:
            continue
        _append_checkpoint_payload_stores_from_shallow_candidate(
            handle,
            stores=stores,
            checked_mapping_ids=checked_mapping_ids,
            payload_mapping_ids=payload_mapping_ids,
        )
        resolver = _handle_resolver(handle)
        _append_checkpoint_payload_stores_from_shallow_candidate(
            resolver,
            stores=stores,
            checked_mapping_ids=checked_mapping_ids,
            payload_mapping_ids=payload_mapping_ids,
        )
    return tuple(stores)


def _iter_from_candidate(candidate: object) -> Iterator[object] | None:
    if isinstance(candidate, Mapping):
        return iter(candidate.values())
    if isinstance(candidate, str | bytes | bytearray):
        return None
    try:
        return iter(candidate) if isinstance(candidate, Iterable) else None
    except TypeError:
        return None


def _iter_linoo_selector_search_children(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield item
        return
    if isinstance(value, CONTAINER_TYPES):
        yield from value
        return

    raw_dict = safe_object_dict(value)
    if raw_dict is not None:
        yield from raw_dict.values()
    yield from iter_object_attribute_values(value)


def _checkpoint_payload_kind(value: object) -> str | None:
    payload_type = qualified_type_name(value)
    if payload_type.endswith(_ANCHOR_PAYLOAD_TYPE_SUFFIX):
        return "anchor"
    if payload_type.endswith(_DELTA_PAYLOAD_TYPE_SUFFIX):
        return "delta"
    return None


def _checkpoint_payload_counts(payloads: Mapping[object, object]) -> tuple[int, int]:
    anchor_count = 0
    delta_count = 0
    for payload in payloads.values():
        payload_kind = _checkpoint_payload_kind(payload)
        if payload_kind == "anchor":
            anchor_count += 1
        elif payload_kind == "delta":
            delta_count += 1
    return anchor_count, delta_count


def _append_checkpoint_payload_store_if_payload_mapping(
    *,
    stores: list[CheckpointPayloadStore],
    checked_mapping_ids: set[int],
    payload_mapping_ids: set[int],
    resolver: object,
    owner: object,
    attr_name: str,
    mapping: Mapping[object, object],
) -> None:
    mapping_id = id(mapping)
    if mapping_id in checked_mapping_ids:
        return
    checked_mapping_ids.add(mapping_id)
    anchor_count, delta_count = _checkpoint_payload_counts(mapping)
    if not (anchor_count or delta_count):
        return
    payload_mapping_ids.add(mapping_id)
    stores.append(
        CheckpointPayloadStore(
            resolver_type=qualified_type_name(resolver),
            resolver_id=id(resolver),
            owner_type=qualified_type_name(owner),
            attr_name=attr_name,
            payloads=mapping,
            anchor_count=anchor_count,
            delta_count=delta_count,
        )
    )


def _append_checkpoint_payload_stores_from_shallow_candidate(
    candidate: object | None,
    *,
    stores: list[CheckpointPayloadStore],
    checked_mapping_ids: set[int],
    payload_mapping_ids: set[int],
) -> None:
    if candidate is None:
        return
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        mapping = raw_getattr(candidate, attr_name)
        if isinstance(mapping, Mapping):
            _append_checkpoint_payload_store_if_payload_mapping(
                stores=stores,
                checked_mapping_ids=checked_mapping_ids,
                payload_mapping_ids=payload_mapping_ids,
                resolver=candidate,
                owner=candidate,
                attr_name=attr_name,
                mapping=mapping,
            )
    owner = raw_getattr(candidate, "owner")
    if owner is None:
        return
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        mapping = raw_getattr(owner, attr_name)
        if isinstance(mapping, Mapping):
            _append_checkpoint_payload_store_if_payload_mapping(
                stores=stores,
                checked_mapping_ids=checked_mapping_ids,
                payload_mapping_ids=payload_mapping_ids,
                resolver=candidate,
                owner=owner,
                attr_name=f"owner.{attr_name}",
                mapping=mapping,
            )


def _handle_resolver(handle: object) -> object | None:
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


def _node_tree_node(node: object) -> object | None:
    return raw_getattr(node, "tree_node")


def _node_tree_evaluation(node: object) -> object | None:
    return raw_getattr(node, "tree_evaluation") or raw_getattr(
        node,
        "node_evaluation",
    )


def _tree_node_slot(node: object, slot_name: str) -> object | None:
    tree_node = _node_tree_node(node)
    if tree_node is None:
        return None
    return raw_getattr(tree_node, slot_name)


def _iter_parent_branch_refs(parent_nodes: object | None) -> Iterator[object]:
    single_parent_branch_keys = raw_getattr(parent_nodes, "branch_keys")
    if single_parent_branch_keys is not None:
        if isinstance(single_parent_branch_keys, Iterable) and not isinstance(
            single_parent_branch_keys,
            str | bytes | bytearray,
        ):
            yield from single_parent_branch_keys
        else:
            yield single_parent_branch_keys
        return

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
    branch = raw_getattr(branches_children, "branch") if branches_children else None
    if branch is not None:
        yield branch
