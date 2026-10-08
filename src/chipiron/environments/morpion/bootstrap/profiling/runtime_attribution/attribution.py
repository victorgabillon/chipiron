"""Bounded component accounting reusing the existing non-materializing size walker."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import chain, islice
from typing import Any

import torch

from chipiron.environments.morpion.bootstrap.pipeline_memory import current_rss_mb
from chipiron.environments.morpion.bootstrap.profiling.recursive.context import (
    materialize_profile_nodes,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    DeepSizeStats,
    deep_size,
    size_or_zero,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.node_evaluation import (
    node_evaluation_runtime_histograms,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    iter_direct_field_entries,
    len_or_none,
    raw_attr_path,
    raw_getattr,
    safe_object_dict,
)


@dataclass(frozen=True)
class ProfileLimits:
    """Hard bounds on retained sample references and recursive visited identities."""

    sample_nodes: int = 128
    max_objects: int = 20_000
    max_depth: int = 6

    def __post_init__(self) -> None:
        """Reject unlimited settings, including negative and excessively large caps."""
        if not (
            1 <= self.sample_nodes <= 512
            and 1 <= self.max_objects <= 100_000
            and 0 <= self.max_depth <= 12
        ):
            message = "Require 1..512 sample nodes, 1..100000 recursive objects and depth 0..12."
            raise ValueError(message)


def _values(value: object) -> Iterator[object]:
    return iter(value.values()) if isinstance(value, Mapping) else iter(())


def _nodes(runtime: object) -> Iterator[object]:
    buckets = raw_attr_path(
        runtime, ("tree", "descendants", "descendants_at_tree_depth")
    )
    for bucket in _values(buckets):
        yield from _values(bucket)


def _unique(values: Iterable[object | None]) -> tuple[object, ...]:
    # Inputs here are bounded samples/direct component roots, never the whole tree.
    return tuple({id(v): v for v in values if v is not None}.values())


def _children(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield item
    elif isinstance(value, list | tuple | set | frozenset):
        yield from value
    else:
        for _, field in iter_direct_field_entries(value):
            yield field


def account_components(
    groups: Mapping[str, tuple[object, ...]],
    *,
    limits: ProfileLimits,
    deep: bool = False,
) -> dict[str, Any]:
    """Partition observed identities once; do not sum overlapping standalone sizes.

    Shallow roots are reserved before recursive traversal. Extra recursive bytes
    use one global visited set and budget, with first-reference attribution. They
    are a bounded reachable lower bound, NOT a proof of exclusive ownership.
    """
    seen: set[int] = set()
    rows: dict[str, dict[str, Any]] = {}
    for name, roots in groups.items():
        byte_count = count = 0
        for root in roots:
            for obj in (root, safe_object_dict(root)):
                if obj is None or id(obj) in seen:
                    continue
                seen.add(id(obj))
                count += 1
                byte_count += size_or_zero(obj)
        rows[name] = {
            "shallow_bytes": byte_count,
            "shallow_object_count": count,
            "root_count": len(roots),
            "extra_reachable_bytes": 0,
            "extra_object_count": 0,
        }
    # Roots have a separate bounded sample budget; max_objects caps additional work.
    stats = DeepSizeStats(max_objects=limits.max_objects)
    if deep:
        for group_index, (name, roots) in enumerate(groups.items()):
            budget = (limits.max_objects - stats.visited_objects) // (
                len(groups) - group_index
            )
            component_stats = DeepSizeStats(max_objects=budget)
            for root in roots:
                for child in _children(root):
                    if component_stats.visited_objects >= budget:
                        component_stats.capped = True
                        break
                    rows[name]["extra_reachable_bytes"] += deep_size(
                        child,
                        seen=seen,
                        stats=component_stats,
                        max_depth=limits.max_depth,
                    )
                if component_stats.visited_objects >= budget:
                    break
            rows[name]["extra_object_count"] = component_stats.visited_objects
            rows[name]["recursive_object_budget"] = budget
            rows[name]["recursive_capped"] = component_stats.capped
            stats.visited_objects += component_stats.visited_objects
            stats.capped |= component_stats.capped
            stats.max_depth_reached_count += component_stats.max_depth_reached_count
            stats.recursion_error_count += component_stats.recursion_error_count
    return {
        "components": rows,
        "recursive": asdict(stats),
        "unique_observed_objects": len(seen),
        "accounted_python_bytes": sum(
            r["shallow_bytes"] + r["extra_reachable_bytes"] for r in rows.values()
        ),
    }


def _selector_roots(selector: object, cap: int) -> tuple[object, ...]:
    """Follow only known wrapper links; never search arbitrary runtime object graphs."""
    values = [selector]
    for _ in range(4):
        nested = raw_getattr(values[-1], "base_selector")
        if nested is None or any(nested is value for value in values):
            break
        values.append(nested)
    roots: list[object] = values[:]
    for item in values:
        fields = safe_object_dict(item)
        if fields is not None:
            roots.extend(islice(fields.values(), cap))
    return _unique(roots)


def _model_storage(master: object) -> dict[str, int]:
    """Count native tensor storage separately, deduplicating parameter/buffer views."""
    model = raw_getattr(master, "regressor")
    totals = {"cpu_bytes": 0, "cuda_bytes": 0, "storage_count": 0}
    if not isinstance(model, torch.nn.Module):
        return totals
    seen: set[tuple[str, int]] = set()
    for tensor in islice(chain(model.parameters(), model.buffers()), 10000):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        totals["storage_count"] += 1
        totals["cuda_bytes" if tensor.is_cuda else "cpu_bytes"] += storage.nbytes()
    return totals


def runtime_profile(
    runner: object, *, limits: ProfileLimits, deep: bool = False
) -> dict[str, Any]:
    """Inspect concrete slots and containers without reading node.state/tag properties."""
    runtime = raw_getattr(runner, "_runtime")
    count = raw_attr_path(runtime, ("tree", "nodes_count"))
    if not isinstance(count, int) or count < 1:
        message = "A restored runtime with a positive node count is required."
        raise ValueError(message)
    stride = max(1, count // limits.sample_nodes)
    nodes = materialize_profile_nodes(
        islice(_nodes(runtime), 0, None, stride), node_cap=limits.sample_nodes
    )
    trees = _unique(raw_getattr(node, "tree_node") for node in nodes)
    evaluations = _unique(raw_getattr(node, "tree_evaluation") for node in nodes)
    handles = _unique(raw_getattr(node, "state_handle_") for node in trees)
    resolvers = _unique(raw_getattr(handle, "resolver") for handle in handles)
    stores = _unique(
        raw_getattr(resolver, "state_payloads_by_node_id") for resolver in resolvers
    )
    payloads: list[object] = []
    for store in stores:
        if isinstance(store, Mapping):
            step = max(1, len(store) // limits.sample_nodes)
            payloads.extend(islice(store.values(), 0, None, step))
            del payloads[limits.sample_nodes :]
    anchors = _unique(
        p for p in payloads if type(p).__name__ == "AnchorCheckpointStatePayload"
    )
    deltas = _unique(
        p for p in payloads if type(p).__name__ == "DeltaCheckpointStatePayload"
    )
    caches = _unique(
        raw_getattr(resolver, "_resolved_states") for resolver in resolvers
    )
    selector = raw_getattr(runtime, "node_selector")
    selector_roots = _selector_roots(selector, limits.sample_nodes)
    sparse_tables = _unique(raw_getattr(s, "_node_state_by_id") for s in selector_roots)
    descendants = raw_attr_path(runtime, ("tree", "descendants"))
    depth_buckets = raw_getattr(descendants, "descendants_at_tree_depth")
    master = raw_attr_path(
        runtime, ("tree_manager", "node_evaluator", "master_state_value_evaluator")
    )
    groups = {
        "algorithm_nodes": _unique(nodes),
        "structural_tree_nodes": trees,
        "node_tree_evaluations": evaluations,
        "child_parent_edges": _unique(
            raw_getattr(n, slot)
            for n in trees
            for slot in ("parent_nodes_", "branches_children_")
        ),
        "branch_opened_unopened": _unique(
            raw_getattr(n, "non_opened_branches_") for n in trees
        ),
        "exploration_indices": _unique(
            raw_getattr(n, "exploration_index_data") for n in nodes
        ),
        "state_handles": handles,
        "checkpoint_resolvers": resolvers,
        "checkpoint_payload_stores": _unique((
            *stores,
            *(
                raw_getattr(s, name)
                for s in stores
                for name in ("payloads_by_dense_node_id", "payloads_by_node_id")
            ),
        )),
        "anchor_payloads_sample": anchors,
        "delta_payloads_sample": deltas,
        "state_summaries_sample": _unique(
            raw_getattr(p, "state_summary") for p in payloads
        ),
        "resolved_state_caches": caches,
        "decoded_states_sample": _unique(
            v for cache in caches for v in islice(_values(cache), limits.sample_nodes)
        ),
        "rematerialization_cache": _unique((
            raw_attr_path(
                runner, ("_live_compact_state_resolver", "_decoded_state_cache")
            ),
        )),
        "linoo_selector_structures": selector_roots,
        "tree_descendant_bookkeeping": _unique((
            raw_getattr(runtime, "tree"),
            descendants,
            depth_buckets,
            *islice(_values(depth_buckets), limits.sample_nodes),
        )),
        "latest_expansions": _unique((raw_getattr(runtime, "latest_tree_expansions"),)),
        "model_evaluator": _unique((master, raw_getattr(master, "regressor"))),
    }
    result = account_components(groups, limits=limits, deep=deep)
    model_storage = _model_storage(master)
    rss = current_rss_mb()
    rss_bytes = None if rss is None else int(rss * 1024**2)
    for name, row in result["components"].items():
        measured = row["shallow_bytes"] + row["extra_reachable_bytes"]
        row["observed_bytes_per_tree_node"] = measured / count
        row["fraction_of_rss"] = None if not rss_bytes else measured / rss_bytes
        row["sample_mean_shallow_bytes_per_node"] = (
            row["shallow_bytes"] / len(nodes)
            if name
            in {
                "algorithm_nodes",
                "structural_tree_nodes",
                "node_tree_evaluations",
                "child_parent_edges",
                "branch_opened_unopened",
                "exploration_indices",
                "state_handles",
            }
            and nodes
            else None
        )
    for row in result["components"].values():
        mean = row["sample_mean_shallow_bytes_per_node"]
        row["projected_shallow_bytes_assuming_unique_per_node"] = (
            None if mean is None else mean * count
        )
    result.update({
        "node_count": count,
        "sample_node_count": len(nodes),
        "sample_stride": stride,
        "sample_method": "systematic depth/insertion-order sample; not random, not a population census",
        "limits": asdict(limits),
        "mode": "bounded_deep" if deep else "cheap_shallow",
        "rss_mib": rss,
        "model_tensor_storage": model_storage,
        "unattributed_rss_bytes": None
        if rss_bytes is None
        else max(
            0, rss_bytes - result["accounted_python_bytes"] - model_storage["cpu_bytes"]
        ),
        "accounting_limitations": "RSS includes baseline imports, allocator slack, native/tensor storage and unsampled objects. Recursive categories are exclusive by visited identity and first reach, not proven ownership. Do not sum extrapolated samples or label the residual allocator waste.",
        "optimizations": {
            "sample_handle_types": dict(Counter(type(h).__name__ for h in handles)),
            "sample_resolver_count": len(resolvers),
            "payload_store_types": [type(s).__name__ for s in stores],
            "payload_store_counts": [len(s) for s in stores if isinstance(s, Mapping)],
            "resolved_state_cache_counts": [
                len(c) for c in caches if isinstance(c, Mapping)
            ],
            "rematerialization_cache_count": len_or_none(
                raw_attr_path(
                    runner, ("_live_compact_state_resolver", "_decoded_state_cache")
                )
            )
            or 0,
            "linoo_sample_default_entries": sum(
                raw_getattr(value, "status") == "opened"
                for table in sparse_tables
                for value in islice(_values(table), limits.sample_nodes)
            ),
            "linoo_sparse_table_counts": [
                len(s) for s in sparse_tables if isinstance(s, Mapping)
            ],
            "lazy_evaluation_sample": node_evaluation_runtime_histograms(nodes),
        },
    })
    return result
