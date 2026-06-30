"""Tree-topology diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sized

from chipiron.environments.morpion.bootstrap.pipeline_memory import format_metric

from .deep_size import DeepSizeStats, deep_size, size_or_zero
from .object_access import len_or_none, qualified_type_name, raw_getattr

_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64

__all__ = [
    "branch_ref_count",
    "child_link_count_from_storage",
    "child_link_storage_detail_histogram",
    "iter_child_branch_refs",
    "iter_parent_branch_refs",
    "node_depth_or_none",
    "node_eval_bool",
    "node_eval_slot",
    "node_id_or_none",
    "node_tree_evaluation",
    "node_tree_node",
    "parent_link_storage_histogram",
    "tree_node_slot",
    "tree_topology_histograms",
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


def _exclusive_shell_size(roots: Iterable[object | None], *, seen: set[int]) -> int:
    total = 0
    for root in roots:
        if root is None:
            continue
        root_id = id(root)
        if root_id in seen:
            continue
        seen.add(root_id)
        total += size_or_zero(root)
    return total


def node_tree_node(node: object) -> object | None:
    """Return the wrapped TreeNode for an AlgorithmNode-like object."""
    return raw_getattr(node, "tree_node")


def node_tree_evaluation(node: object) -> object | None:
    """Return the wrapped node evaluation object when present."""
    return raw_getattr(node, "tree_evaluation") or raw_getattr(
        node,
        "node_evaluation",
    )


def tree_node_slot(node: object, slot_name: str) -> object | None:
    """Return a raw slot from a wrapped TreeNode."""
    tree_node = node_tree_node(node)
    if tree_node is None:
        return None
    return raw_getattr(tree_node, slot_name)


def node_eval_slot(node: object, slot_name: str) -> object | None:
    """Return a raw slot from a wrapped node evaluation."""
    node_eval = node_tree_evaluation(node)
    if node_eval is None:
        return None
    return raw_getattr(node_eval, slot_name)


def node_id_or_none(node: object) -> int | None:
    """Return a stable integer node id, when the object exposes one."""
    for attr_name in ("id", "id_"):
        value = raw_getattr(node, attr_name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    tree_node = node_tree_node(node)
    if tree_node is not None:
        return node_id_or_none(tree_node)
    return None


def node_depth_or_none(node: object) -> int | None:
    """Return a cheap tree-depth value when present."""
    tree_node = node_tree_node(node) or node
    for attr_name in ("tree_depth_", "tree_depth", "depth"):
        value = raw_getattr(tree_node, attr_name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _safe_bool_method(value: object | None, method_name: str) -> bool | None:
    if value is None:
        return None
    method = raw_getattr(value, method_name)
    if not callable(method):
        return None
    try:
        return bool(method())
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def node_eval_bool(node: object, method_name: str) -> bool | None:
    """Call a boolean node-evaluation method when it is safely available."""
    return _safe_bool_method(node_tree_evaluation(node), method_name)


def iter_parent_branch_refs(parent_nodes: object | None) -> Iterator[object]:
    """Yield branch-key references from raw parent-link storage."""
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


def iter_child_branch_refs(branches_children: object | None) -> Iterator[object]:
    """Yield branch-key references from raw child-link storage."""
    if isinstance(branches_children, Mapping):
        yield from branches_children.keys()
        return
    branch = raw_getattr(branches_children, "branch") if branches_children else None
    if branch is not None:
        yield branch


def tree_topology_histograms(nodes: Iterable[object]) -> dict[str, object]:
    """Return topology histograms without materializing child/parent properties."""
    child_link_count_hist = Counter[str]()
    parent_count_hist = Counter[str]()
    total_parent_branch_refs_hist = Counter[str]()
    branches_children_type_hist = Counter[str]()
    parent_nodes_type_hist = Counter[str]()
    non_opened_branches_type_hist = Counter[str]()

    for node in nodes:
        tree_node = node_tree_node(node) or node
        branches_children = raw_getattr(tree_node, "branches_children_")
        parent_nodes = raw_getattr(tree_node, "parent_nodes_")
        non_opened_branches = raw_getattr(tree_node, "non_opened_branches_")

        branches_children_type_hist[qualified_type_name(branches_children)] += 1
        parent_nodes_type_hist[qualified_type_name(parent_nodes)] += 1
        non_opened_branches_type_hist[qualified_type_name(non_opened_branches)] += 1

        child_count = child_link_count_from_storage(branches_children)
        child_link_count_hist[str(child_count)] += 1

        parent_count = _parent_storage_parent_count(parent_nodes)
        parent_count_hist[str(parent_count)] += 1
        total_parent_refs = _parent_storage_branch_ref_count(parent_nodes)
        total_parent_branch_refs_hist[str(total_parent_refs)] += 1

    return {
        "child_link_count": dict(child_link_count_hist),
        "parent_nodes_len": dict(parent_count_hist),
        "parent_branch_refs": dict(total_parent_branch_refs_hist),
        "branches_children_types": dict(branches_children_type_hist),
        "parent_nodes_types": dict(parent_nodes_type_hist),
        "non_opened_branches_types": dict(non_opened_branches_type_hist),
    }


def parent_link_storage_histogram(
    nodes: Iterable[object],
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
) -> dict[str, object]:
    """Return focused diagnostics for raw parent-link storage."""
    node_count = 0
    zero_parent_nodes = 0
    one_parent_nodes = 0
    multi_parent_nodes = 0
    single_parent_storages: list[object] = []
    multi_parent_storages: list[object] = []
    parent_storage_type_counts = Counter[str]()
    parent_branch_ref_count = 0
    parent_storage_shallow_bytes = 0

    for node in nodes:
        node_count += 1
        tree_node = node_tree_node(node) or node
        parent_nodes = raw_getattr(tree_node, "parent_nodes_")
        parent_storage_type_counts[qualified_type_name(parent_nodes)] += 1
        parent_storage_shallow_bytes += size_or_zero(parent_nodes)

        parent_count = _parent_storage_parent_count(parent_nodes)
        if parent_count == 0:
            zero_parent_nodes += 1
        elif parent_count == 1:
            one_parent_nodes += 1
            single_parent_storages.append(parent_nodes)
        else:
            multi_parent_nodes += 1
            multi_parent_storages.append(parent_nodes)

        parent_branch_ref_count += _parent_storage_branch_ref_count(parent_nodes)

    single_stats = DeepSizeStats(max_objects=max_objects)
    single_parent_recursive_bytes = _exclusive_deep_size(
        single_parent_storages,
        seen=set(),
        max_depth=max_depth,
        stats=single_stats,
    )
    multi_stats = DeepSizeStats(max_objects=max_objects)
    multi_parent_recursive_bytes = _exclusive_deep_size(
        multi_parent_storages,
        seen=set(),
        max_depth=max_depth,
        stats=multi_stats,
    )
    total_parent_recursive_bytes = (
        single_parent_recursive_bytes + multi_parent_recursive_bytes
    )

    return {
        "node_count": node_count,
        "zero_parent_node_count": zero_parent_nodes,
        "one_parent_node_count": one_parent_nodes,
        "multi_parent_node_count": multi_parent_nodes,
        "parent_branch_ref_count": parent_branch_ref_count,
        "parent_storage_types": dict(
            _ordered_counter_items(parent_storage_type_counts)
        ),
        "parent_storage_shallow_bytes": parent_storage_shallow_bytes,
        "single_parent_recursive_bytes": single_parent_recursive_bytes,
        "single_parent_recursive_capped": single_stats.capped,
        "multi_parent_recursive_bytes": multi_parent_recursive_bytes,
        "multi_parent_recursive_capped": multi_stats.capped,
        "average_parent_link_recursive_bytes_per_node": format_metric(
            total_parent_recursive_bytes / node_count if node_count else None
        ),
    }


def _parent_storage_parent_count(parent_nodes: object | None) -> int:
    if parent_nodes is None:
        return 0
    if raw_getattr(parent_nodes, "parent_node") is not None:
        return 1
    if isinstance(parent_nodes, Mapping):
        return len(parent_nodes)
    return 0


def branch_ref_count(branch_refs: object | None) -> int:
    """Return the count of branch references represented by raw storage."""
    if branch_refs is None:
        return 0
    branch_ref_len = len_or_none(branch_refs)
    return 1 if branch_ref_len is None else branch_ref_len


def _parent_storage_branch_ref_count(parent_nodes: object | None) -> int:
    branch_keys = raw_getattr(parent_nodes, "branch_keys")
    if branch_keys is not None:
        return branch_ref_count(branch_keys)
    if not isinstance(parent_nodes, Mapping):
        return 0
    return sum(branch_ref_count(branch_set) for branch_set in parent_nodes.values())


def child_link_count_from_storage(branches_children: object | None) -> int:
    """Return the count of child edges represented by raw storage."""
    if branches_children is None:
        return 0
    if isinstance(branches_children, Mapping):
        return len(branches_children)
    if raw_getattr(branches_children, "branch") is not None:
        return 1
    if isinstance(branches_children, tuple) and len(branches_children) == 2:
        return 1
    if isinstance(branches_children, Sized):
        return len(branches_children)
    return 1


def child_link_storage_detail_histogram(
    nodes: Iterable[object],
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
    max_objects: int | None = None,
    branch_key_sample_cap: int = 5_000,
) -> dict[str, object]:
    """Return focused diagnostics for raw child-link storage."""
    node_count = 0
    nodes_with_no_children = 0
    nodes_with_children = 0
    total_child_edges = 0
    child_container_type_counts = Counter[str]()
    branch_key_type_counts = Counter[str]()
    child_node_ref_count = 0
    child_link_containers: list[object] = []
    sampled_branch_keys: list[object] = []
    sampled_branch_key_ids: set[int] = set()
    child_branch_key_ids: set[int] = set()
    parent_branch_key_ids: set[int] = set()

    for node in nodes:
        node_count += 1
        tree_node = node_tree_node(node) or node
        branches_children = raw_getattr(tree_node, "branches_children_")
        parent_nodes = raw_getattr(tree_node, "parent_nodes_")
        child_container_type_counts[qualified_type_name(branches_children)] += 1
        if branches_children is not None:
            child_link_containers.append(branches_children)

        child_count = child_link_count_from_storage(branches_children)
        total_child_edges += child_count
        if child_count == 0:
            nodes_with_no_children += 1
        else:
            nodes_with_children += 1

        for branch_key in iter_child_branch_refs(branches_children):
            child_branch_key_ids.add(id(branch_key))
            branch_key_type_counts[qualified_type_name(branch_key)] += 1
            if (
                len(sampled_branch_keys) < branch_key_sample_cap
                and id(branch_key) not in sampled_branch_key_ids
            ):
                sampled_branch_key_ids.add(id(branch_key))
                sampled_branch_keys.append(branch_key)

        if isinstance(branches_children, Mapping):
            child_node_ref_count += len(branches_children)
        else:
            child_ref = raw_getattr(branches_children, "child_node")
            if child_ref is None:
                child_ref = raw_getattr(branches_children, "child")
            if child_ref is not None:
                child_node_ref_count += 1

        for branch_key in iter_parent_branch_refs(parent_nodes):
            parent_branch_key_ids.add(id(branch_key))

    container_stats = DeepSizeStats(max_objects=max_objects)
    child_link_container_recursive_bytes = _exclusive_deep_size(
        child_link_containers,
        seen=set(),
        max_depth=max_depth,
        stats=container_stats,
    )
    branch_key_stats = DeepSizeStats(max_objects=max_objects)
    branch_key_sample_recursive_bytes = _exclusive_deep_size(
        sampled_branch_keys,
        seen=set(),
        max_depth=max_depth,
        stats=branch_key_stats,
    )
    branch_key_sample_shallow_bytes = _exclusive_shell_size(
        sampled_branch_keys,
        seen=set(),
    )
    duplicate_branch_key_ref_count = len(child_branch_key_ids & parent_branch_key_ids)

    return {
        "node_count_scanned": node_count,
        "nodes_with_no_children": nodes_with_no_children,
        "nodes_with_children": nodes_with_children,
        "total_child_edges": total_child_edges,
        "average_children_per_non_empty_node": (
            None
            if nodes_with_children == 0
            else format_metric(total_child_edges / nodes_with_children)
        ),
        "child_container_type_counts": dict(
            _ordered_counter_items(child_container_type_counts)
        ),
        "child_link_container_recursive_bytes": child_link_container_recursive_bytes,
        "child_link_container_recursive_visited_objects": (
            container_stats.visited_objects
        ),
        "child_link_container_recursive_capped": container_stats.capped,
        "child_link_container_recursive_max_depth_reached_count": (
            container_stats.max_depth_reached_count
        ),
        "child_link_container_recursive_recursion_error_count": (
            container_stats.recursion_error_count
        ),
        "branch_key_sample_count": len(sampled_branch_keys),
        "branch_key_sample_cap": branch_key_sample_cap,
        "branch_key_sample_shallow_bytes": branch_key_sample_shallow_bytes,
        "branch_key_sample_recursive_bytes": branch_key_sample_recursive_bytes,
        "branch_key_sample_recursive_visited_objects": branch_key_stats.visited_objects,
        "branch_key_sample_recursive_capped": branch_key_stats.capped,
        "child_node_reference_count": child_node_ref_count,
        "top_branch_key_python_types": dict(
            _ordered_counter_items(branch_key_type_counts)
        ),
        "stores_tuple_branch_keys": any(
            type_name == "tuple" for type_name in branch_key_type_counts
        ),
        "stores_list_branch_keys": any(
            type_name == "list" for type_name in branch_key_type_counts
        ),
        "stores_int_branch_keys": any(
            type_name == "int" for type_name in branch_key_type_counts
        ),
        "stores_dict_branch_keys": any(
            type_name == "dict" for type_name in branch_key_type_counts
        ),
        "duplicate_branch_key_refs_with_parent_links": duplicate_branch_key_ref_count,
        "duplicate_storage_with_parent_links_detectable": (
            duplicate_branch_key_ref_count > 0
        ),
    }
