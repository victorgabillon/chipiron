"""Recursive restored-tree memory diagnostics for Morpion growth runtimes."""
# pylint: disable=too-many-lines

from __future__ import annotations

import gc
import logging
import time
from collections import Counter
from collections.abc import (  # pylint: disable=unused-import
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sized,
)
from dataclasses import dataclass, field
from enum import Enum
from types import FrameType
from typing import cast

from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    current_rss_mb,
    format_metric,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.context import (
    CheckpointPayloadStore,
    ComponentProfileRecord,
    RecursiveProfileContext,
    build_recursive_profile_context,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.context import (
    component_roots as _component_roots,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.context import (
    find_linoo_selector_root as _find_linoo_selector_root,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    DeepSizeStats,
    deep_size,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    deep_size_stats_capped as _deep_size_stats_capped,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    mb as _mb,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    measure_standalone_reachable as _measure_standalone_reachable,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.deep_size import (
    size_or_zero as _size_or_zero,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    ATOMIC_TYPES as _ATOMIC_TYPES,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    CONTAINER_TYPES as _CONTAINER_TYPES,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    CONTAINER_VALUE_TYPES as _CONTAINER_VALUE_TYPES,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    iter_direct_field_entries as _iter_direct_field_entries,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    iter_object_attribute_values as _iter_object_attribute_values,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    len_or_none as _len_or_none,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    qualified_type_name as _qualified_type_name,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    raw_getattr as _raw_getattr,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    raw_getattr_present as _raw_getattr_present,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    safe_object_dict as _safe_object_dict,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    should_skip_deep as _should_skip_deep,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    slot_names,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive.object_access import (
    small_len_bucket as _small_len_bucket,
)

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

_DEFAULT_CHECKPOINT_HANDLE_SCAN_CAP = 50_000
_DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP = 100
_DEFAULT_RECURSIVE_PROFILE_MAX_OBJECTS = 3_000_000
_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_DEFAULT_LINOO_NODE_STATE_SAMPLE_CAP = 5_000
_DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP = 5_000
_DEFAULT_FROZENSET_OWNER_REFERRER_SCAN_CAP = 16
_DEFAULT_PAYLOAD_SHAPE_TOP_N = 8
_DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS = 4
_ANCHOR_PAYLOAD_TYPE_SUFFIX = "AnchorCheckpointStatePayload"
_DELTA_PAYLOAD_TYPE_SUFFIX = "DeltaCheckpointStatePayload"
_PROJECT_TYPE_PREFIXES = ("anemone.", "chipiron.", "atomheart.", "valanga.")
_FROZENSET_MORPION_STATE_FIELDS = (
    "points",
    "used_unit_segments",
    "played_moves",
)
_TRACKED_SHALLOW_TYPE_SUFFIXES = (
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "BranchOrderingKey",
    "Value",
    "AlgorithmNode",
    "TreeNode",
    "NodeMaxEvaluation",
    "CheckpointBackedStateHandle",
    "_LinooNodeState",
    "AnchorCheckpointStatePayload",
    "DeltaCheckpointStatePayload",
)
_KNOWN_CHECKPOINT_STORE_ATTR_NAMES = (
    "state_payloads_by_node_id",
    "payloads",
    "_state_payloads_by_node_id",
    "_payloads",
    "payloads_by_node_id",
    "_payloads_by_node_id",
)
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
_RUNTIME_STATE_SLOT_LABELS: dict[str, str] = {
    "decision_ordering_": "DecisionOrderingState",
    "pv_state_": "PrincipalVariationState",
    "branch_frontier_": "BranchFrontierState",
    "backup_runtime_": "Top2ExactnessPvRuntime",
}
_KNOWN_CHECKPOINT_CANDIDATE_PATHS: tuple[tuple[str, ...], ...] = tuple(
    dict.fromkeys(
        (
            *_CHECKPOINT_ROOT_ATTR_PATHS,
            ("checkpoint_state_resolver",),
            ("_checkpoint_state_resolver",),
            ("_runtime", "checkpoint_state_resolver"),
            ("_runtime", "_checkpoint_state_resolver"),
            ("runtime", "checkpoint_state_resolver"),
            ("runtime", "_checkpoint_state_resolver"),
        )
    )
)

type ProfileRoot = tuple[str, object]


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


@dataclass(slots=True)
class _FrozensetOwnershipAccumulator:
    """Mutable counters for one shallow frozenset ownership scan."""

    total_count: int = 0
    total_shallow_bytes: int = 0
    len_bucket_counts: Counter[str] = field(default_factory=Counter)
    sampled_frozensets: list[frozenset[object]] = field(default_factory=list)


@dataclass(slots=True)
class _MorpionStateFrozensetAccumulator:
    """Mutable counters for direct MorpionState frozenset field scans."""

    morpion_state_count: int = 0
    field_ref_counts: Counter[str] = field(default_factory=Counter)
    field_len_bucket_counts: Counter[str] = field(default_factory=Counter)
    total_field_shallow_bytes: int = 0


def _frozenset_len_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 9:
        return "5-9"
    if length <= 24:
        return "10-24"
    if length <= 49:
        return "25-49"
    if length <= 99:
        return "50-99"
    if length <= 199:
        return "100-199"
    return "200+"


def _format_name_float_pairs(items: Iterable[tuple[str, float]]) -> str:
    return (
        "[" + ",".join(f"{name}:{format_metric(value)}" for name, value in items) + "]"
    )


def _format_name_int_pairs(items: Iterable[tuple[str, int]]) -> str:
    return "[" + ",".join(f"{name}:{value}" for name, value in items) + "]"


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


def _resolve_linoo_selector(
    selector: object | None,
) -> tuple[object | None, Mapping[object, object] | None]:
    if selector is None:
        return None, None
    linoo_selector = _find_linoo_selector_root(selector)
    if linoo_selector is None:
        return None, None
    node_state_by_id = _raw_getattr(linoo_selector, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
    if not isinstance(node_state_by_id, Mapping):
        return linoo_selector, None
    return linoo_selector, node_state_by_id


def _linoo_slot_value_kind(value: object) -> str:
    if value is None:
        return "None"
    type_name = _qualified_type_name(value)
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
        type_name = _qualified_type_name(value)
        if any(type_name.endswith(suffix) for suffix in suffixes):
            return True
        if isinstance(value, _ATOMIC_TYPES) or _should_skip_deep(value):
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                stack.append(key)
                stack.append(item)
            continue
        if isinstance(value, _CONTAINER_TYPES):
            stack.extend(value)
            continue
        stack.extend(_iter_object_attribute_values(value))
    return False


def _tree_reachability_flags(root: object) -> dict[str, bool]:
    return {
        "reaches_algorithm_node": _object_reaches_type_suffix(
            root,
            ("AlgorithmNode",),
        ),
        "reaches_tree_node": _object_reaches_type_suffix(root, ("TreeNode",)),
    }


def _observe_frozenset(
    accumulator: _FrozensetOwnershipAccumulator,
    value: frozenset[object],
    *,
    sample_cap: int,
) -> None:
    accumulator.total_count += 1
    accumulator.total_shallow_bytes += _size_or_zero(value)
    accumulator.len_bucket_counts[_frozenset_len_bucket(len(value))] += 1
    if len(accumulator.sampled_frozensets) < sample_cap:
        accumulator.sampled_frozensets.append(value)


def _observe_morpion_state_frozenset_fields(
    accumulator: _MorpionStateFrozensetAccumulator,
    state: object,
) -> None:
    accumulator.morpion_state_count += 1
    for field_name in _FROZENSET_MORPION_STATE_FIELDS:
        field_value = _raw_getattr(state, field_name)
        if not isinstance(field_value, frozenset):
            continue
        accumulator.field_ref_counts[field_name] += 1
        accumulator.field_len_bucket_counts[
            _frozenset_len_bucket(len(field_value))
        ] += 1
        accumulator.total_field_shallow_bytes += _size_or_zero(field_value)


def _is_morpion_state_type_name(type_name: str) -> bool:
    return type_name == "MorpionState" or type_name.endswith(".MorpionState")


def _morpion_state_field_ref(
    referrer: object,
    target: frozenset[object],
) -> str | None:
    owner_dict = _safe_object_dict(referrer)
    if owner_dict is not None:
        for field_name in _FROZENSET_MORPION_STATE_FIELDS:
            if owner_dict.get(field_name) is target:
                return field_name
    for slot_name in slot_names(referrer):
        if slot_name not in _FROZENSET_MORPION_STATE_FIELDS:
            continue
        if _raw_getattr(referrer, slot_name) is target:
            return slot_name
    return None


def _is_skippable_owner_referrer(value: object) -> bool:
    return isinstance(value, FrameType) or _should_skip_deep(value)


def _morpion_state_field_ref_via_owner_dict(
    referrer: Mapping[object, object],
    target: frozenset[object],
    *,
    ignored_referrer_ids: set[int],
    owner_referrer_scan_cap: int,
) -> tuple[int, str] | None:
    owners_scanned = 0
    for owner in gc.get_referrers(referrer):
        if id(owner) in ignored_referrer_ids:
            continue
        if _is_skippable_owner_referrer(owner):
            continue
        owners_scanned += 1
        if owners_scanned > owner_referrer_scan_cap:
            break
        owner_type_name = _qualified_type_name(owner)
        if not _is_morpion_state_type_name(owner_type_name):
            continue
        owner_dict = _safe_object_dict(owner)
        if owner_dict is not referrer:
            continue
        field_name = _morpion_state_field_ref(owner, target)
        if field_name is not None:
            return (id(owner), field_name)
    return None


def _morpion_state_owner_field_ref(
    referrer: object,
    target: frozenset[object],
    *,
    ignored_referrer_ids: set[int],
    owner_referrer_scan_cap: int,
) -> tuple[int, str] | None:
    referrer_type_name = _qualified_type_name(referrer)
    if _is_morpion_state_type_name(referrer_type_name):
        field_name = _morpion_state_field_ref(referrer, target)
        if field_name is not None:
            return (id(referrer), field_name)
        return None
    if isinstance(referrer, dict):
        return _morpion_state_field_ref_via_owner_dict(
            referrer,
            target,
            ignored_referrer_ids=ignored_referrer_ids,
            owner_referrer_scan_cap=owner_referrer_scan_cap,
        )
    return None


def _finalize_frozenset_ownership_histogram(
    accumulator: _FrozensetOwnershipAccumulator,
    *,
    ignored_referrer_ids: set[int],
    sample_cap: int,
    top_n: int,
    owner_referrer_scan_cap: int,
) -> dict[str, object]:
    referrer_type_counts = Counter[str]()
    morpion_state_field_refs = Counter[str]()

    base_ignored_referrer_ids = set(ignored_referrer_ids)
    base_ignored_referrer_ids.update(
        {
            id(accumulator),
            id(accumulator.len_bucket_counts),
            id(accumulator.sampled_frozensets),
            id(base_ignored_referrer_ids),
        }
    )

    for frozen_set in accumulator.sampled_frozensets:
        iteration_ignored_referrer_ids = set(base_ignored_referrer_ids)
        seen_owner_field_refs: set[tuple[int, str]] = set()
        for referrer in gc.get_referrers(frozen_set):
            if id(referrer) in iteration_ignored_referrer_ids:
                continue
            if _is_skippable_owner_referrer(referrer):
                continue
            referrer_type_name = _qualified_type_name(referrer)
            referrer_type_counts[referrer_type_name] += 1
            owner_field_ref = _morpion_state_owner_field_ref(
                referrer,
                frozen_set,
                ignored_referrer_ids=iteration_ignored_referrer_ids,
                owner_referrer_scan_cap=owner_referrer_scan_cap,
            )
            if owner_field_ref is None or owner_field_ref in seen_owner_field_refs:
                continue
            seen_owner_field_refs.add(owner_field_ref)
            morpion_state_field_refs[owner_field_ref[1]] += 1

    return {
        "total_count": accumulator.total_count,
        "total_shallow_bytes": accumulator.total_shallow_bytes,
        "len_buckets": _ordered_counter_items(
            accumulator.len_bucket_counts,
            order=(
                "0",
                "1",
                "2-4",
                "5-9",
                "10-24",
                "25-49",
                "50-99",
                "100-199",
                "200+",
            ),
        ),
        "sample_count": len(accumulator.sampled_frozensets),
        "sample_cap": sample_cap,
        "top_referrer_types": referrer_type_counts.most_common(top_n),
        "morpion_state_field_refs": _ordered_counter_items(
            morpion_state_field_refs,
            order=_FROZENSET_MORPION_STATE_FIELDS,
        ),
    }


def frozenset_ownership_histogram(
    *,
    objects: Iterable[object] | None = None,
    sample_cap: int = _DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP,
    top_n: int = 20,
    owner_referrer_scan_cap: int = _DEFAULT_FROZENSET_OWNER_REFERRER_SCAN_CAP,
    ignored_referrer_ids: Iterable[int] = (),
) -> dict[str, object]:
    """Return a bounded ownership sketch for tracked frozensets."""
    gc_objects = gc.get_objects() if objects is None else objects
    accumulator = _FrozensetOwnershipAccumulator()
    effective_sample_cap = max(0, sample_cap)

    for value in gc_objects:
        if isinstance(value, frozenset):
            _observe_frozenset(
                accumulator,
                cast("frozenset[object]", value),
                sample_cap=effective_sample_cap,
            )

    effective_ignored_referrer_ids = set(ignored_referrer_ids)
    if objects is None:
        effective_ignored_referrer_ids.add(id(gc_objects))
    return _finalize_frozenset_ownership_histogram(
        accumulator,
        ignored_referrer_ids=effective_ignored_referrer_ids,
        sample_cap=effective_sample_cap,
        top_n=top_n,
        owner_referrer_scan_cap=max(0, owner_referrer_scan_cap),
    )


def _direct_frozenset_ownership_summary(
    frozenset_ownership: _FrozensetOwnershipAccumulator,
    morpion_state_fields: _MorpionStateFrozensetAccumulator,
) -> dict[str, object]:
    return {
        "total_count": frozenset_ownership.total_count,
        "total_shallow_bytes": frozenset_ownership.total_shallow_bytes,
        "len_buckets": _ordered_counter_items(
            frozenset_ownership.len_bucket_counts,
            order=(
                "0",
                "1",
                "2-4",
                "5-9",
                "10-24",
                "25-49",
                "50-99",
                "100-199",
                "200+",
            ),
        ),
        "morpion_state_count": morpion_state_fields.morpion_state_count,
        "morpion_state_field_refs": _ordered_counter_items(
            morpion_state_fields.field_ref_counts,
            order=_FROZENSET_MORPION_STATE_FIELDS,
        ),
        "morpion_state_field_shallow_bytes": (
            morpion_state_fields.total_field_shallow_bytes
        ),
        "morpion_state_field_len_buckets": _ordered_counter_items(
            morpion_state_fields.field_len_bucket_counts,
            order=(
                "0",
                "1",
                "2-4",
                "5-9",
                "10-24",
                "25-49",
                "50-99",
                "100-199",
                "200+",
            ),
        ),
    }


def _checkpoint_payload_kind(value: object) -> str | None:
    payload_type = _qualified_type_name(value)
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


def _node_state_handle(node: object) -> object | None:
    handle = _tree_node_slot(node, "state_handle_")
    if handle is not None:
        return handle
    tree_node = _node_tree_node(node) or node
    return _raw_getattr(tree_node, "state_handle_") or _raw_getattr(
        tree_node,
        "state_handle",
    )


def _state_from_materialized_handle(handle: object | None) -> object | None:
    if handle is None:
        return None
    return _raw_getattr(handle, "state_")


def _state_handle_storage_kind(handle: object | None) -> str:
    type_name = _qualified_type_name(handle)
    if type_name.endswith("MaterializedStateHandle"):
        return "MaterializedStateHandle"
    if type_name.endswith("CheckpointBackedStateHandle"):
        return "CheckpointBackedStateHandle"
    if handle is None:
        return "None"
    return "other"


def _node_id_or_none(node: object) -> int | None:
    for attr_name in ("id", "id_"):
        value = _raw_getattr(node, attr_name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    tree_node = _node_tree_node(node)
    if tree_node is not None:
        return _node_id_or_none(tree_node)
    return None


def _node_depth_or_none(node: object) -> int | None:
    tree_node = _node_tree_node(node) or node
    for attr_name in ("tree_depth_", "tree_depth", "depth"):
        value = _raw_getattr(tree_node, attr_name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _safe_bool_method(value: object | None, method_name: str) -> bool | None:
    if value is None:
        return None
    method = _raw_getattr(value, method_name)
    if not callable(method):
        return None
    try:
        return bool(cast("Callable[[], object]", method)())
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _node_eval_bool(node: object, method_name: str) -> bool | None:
    return _safe_bool_method(_node_tree_evaluation(node), method_name)


def _selector_node_status_from_table(
    node_state_by_id: Mapping[object, object] | None,
    node: object,
) -> str | None:
    if node_state_by_id is None:
        return None
    node_id = _node_id_or_none(node)
    if node_id is None:
        return None
    state = node_state_by_id.get(node_id)
    if state is None:
        return "opened"
    status = _raw_getattr(state, "status")
    return status if isinstance(status, str) else str(status)


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


def _iter_parent_branch_refs(parent_nodes: object | None) -> Iterator[object]:
    single_parent_branch_keys = _raw_getattr(parent_nodes, "branch_keys")
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
    branch = _raw_getattr(branches_children, "branch") if branches_children else None
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
        tree_node = _node_tree_node(node) or node
        branches_children = _raw_getattr(tree_node, "branches_children_")
        parent_nodes = _raw_getattr(tree_node, "parent_nodes_")
        non_opened_branches = _raw_getattr(tree_node, "non_opened_branches_")

        branches_children_type_hist[_qualified_type_name(branches_children)] += 1
        parent_nodes_type_hist[_qualified_type_name(parent_nodes)] += 1
        non_opened_branches_type_hist[_qualified_type_name(non_opened_branches)] += 1

        child_count = _child_link_count_from_storage(branches_children)
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
        tree_node = _node_tree_node(node) or node
        parent_nodes = _raw_getattr(tree_node, "parent_nodes_")
        parent_storage_type_counts[_qualified_type_name(parent_nodes)] += 1
        parent_storage_shallow_bytes += _size_or_zero(parent_nodes)

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
    if _raw_getattr(parent_nodes, "parent_node") is not None:
        return 1
    if isinstance(parent_nodes, Mapping):
        return len(parent_nodes)
    return 0


def _branch_ref_count(branch_refs: object | None) -> int:
    if branch_refs is None:
        return 0
    branch_ref_len = _len_or_none(branch_refs)
    return 1 if branch_ref_len is None else branch_ref_len


def _parent_storage_branch_ref_count(parent_nodes: object | None) -> int:
    branch_keys = _raw_getattr(parent_nodes, "branch_keys")
    if branch_keys is not None:
        return _branch_ref_count(branch_keys)
    if not isinstance(parent_nodes, Mapping):
        return 0
    return sum(_branch_ref_count(branch_set) for branch_set in parent_nodes.values())


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
        tree_node = _node_tree_node(node) or node
        branches_children = _raw_getattr(tree_node, "branches_children_")
        parent_nodes = _raw_getattr(tree_node, "parent_nodes_")
        child_container_type_counts[_qualified_type_name(branches_children)] += 1
        if branches_children is not None:
            child_link_containers.append(branches_children)

        child_count = _child_link_count_from_storage(branches_children)
        total_child_edges += child_count
        if child_count == 0:
            nodes_with_no_children += 1
        else:
            nodes_with_children += 1

        for branch_key in _iter_child_branch_refs(branches_children):
            child_branch_key_ids.add(id(branch_key))
            branch_key_type_counts[_qualified_type_name(branch_key)] += 1
            if (
                len(sampled_branch_keys) < branch_key_sample_cap
                and id(branch_key) not in sampled_branch_key_ids
            ):
                sampled_branch_key_ids.add(id(branch_key))
                sampled_branch_keys.append(branch_key)

        if isinstance(branches_children, Mapping):
            child_node_ref_count += len(branches_children)
        else:
            child_ref = _raw_getattr(branches_children, "child_node")
            if child_ref is None:
                child_ref = _raw_getattr(branches_children, "child")
            if child_ref is not None:
                child_node_ref_count += 1

        for branch_key in _iter_parent_branch_refs(parent_nodes):
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
        handle_type = _qualified_type_name(handle)
        handle_kind = _state_handle_storage_kind(handle)
        handle_type_counts[handle_type] += 1
        handle_storage_kind_counts[handle_kind] += 1
        if handle_kind == "CheckpointBackedStateHandle":
            checkpoint_backed_state_count += 1

        state = _state_from_materialized_handle(handle)
        if state is None:
            continue
        materialized_state_count += 1
        state_type = _qualified_type_name(state)
        state_type_counts[state_type] += 1
        if _is_morpion_state_type_name(state_type):
            materialized_morpion_state_count += 1
        if id(state) not in materialized_state_ids:
            materialized_state_ids.add(id(state))
            materialized_states.append(state)
        for field_name, field_value in _iter_direct_field_entries(state):
            field_type_counts[f"{field_name}:{_qualified_type_name(field_value)}"] += 1
            field_shallow_bytes[field_name] += _size_or_zero(field_value)
            if isinstance(field_value, frozenset):
                frozenset_field_counts[field_name] += 1
                frozenset_field_shallow_bytes[field_name] += _size_or_zero(field_value)

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
    _linoo_selector, node_state_by_id = _resolve_linoo_selector(selector)
    del _linoo_selector

    for node in nodes:
        node_count += 1
        tree_node = _node_tree_node(node) or node
        depth = _node_depth_or_none(node)
        depth_bucket = "unknown" if depth is None else str(depth)
        depth_counts[depth_bucket] += 1

        handle = _node_state_handle(node)
        has_materialized_state = _state_from_materialized_handle(handle) is not None
        if has_materialized_state:
            materialized_state_count += 1
            materialized_depth_counts[depth_bucket] += 1

        child_count = _child_link_count_from_storage(
            _raw_getattr(tree_node, "branches_children_")
        )
        has_children = child_count > 0
        unopened_branch_count = _branch_ref_count(
            _raw_getattr(tree_node, "non_opened_branches_")
        )
        has_no_unopened_branches = unopened_branch_count == 0
        all_branches_generated = bool(_raw_getattr(tree_node, "all_branches_generated"))
        terminal = _node_eval_bool(node, "is_terminal")
        exact = _node_eval_bool(node, "has_exact_value")
        selector_status = _selector_node_status_from_table(node_state_by_id, node)
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
    profile = _raw_getattr(runner, "profile_state_eviction_runtime")
    if callable(profile):
        try:
            payload = cast("Callable[[], object]", profile)()
        except Exception:  # pylint: disable=broad-exception-caught
            payload = None
        if isinstance(payload, Mapping):
            return {"present": True, **dict(payload)}
    metrics = _raw_getattr(runner, "_state_eviction_metrics")
    snapshot = _raw_getattr(metrics, "snapshot")
    if callable(snapshot):
        try:
            payload = cast("Callable[[], object]", snapshot)()
        except Exception:  # pylint: disable=broad-exception-caught
            payload = None
        if isinstance(payload, Mapping):
            return {"present": True, **dict(payload)}
    return {"present": False}


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
        node_eval = _node_tree_evaluation(node)
        if node_eval is None:
            continue
        for slot_name in _NODE_EVALUATION_RUNTIME_SLOTS:
            runtime_state = _raw_getattr(node_eval, slot_name)
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
            state_type_counts[_qualified_type_name(state)] += 1
            state_is_empty = True
            for field_name, field_value in _iter_direct_field_entries(state):
                field_type_counts[
                    f"{field_name}:{_qualified_type_name(field_value)}"
                ] += 1
                field_shallow_bytes[field_name] += _size_or_zero(field_value)
                field_len = _len_or_none(field_value)
                if field_len is not None:
                    field_len_buckets[
                        f"{field_name}:{_small_len_bucket(field_len)}"
                    ] += 1
                    if field_len > 0:
                        state_is_empty = False
                elif field_value not in (None, False, 0):
                    state_is_empty = False

                if isinstance(field_value, Mapping):
                    top_child_type_counts.update(
                        _qualified_type_name(item) for item in field_value.values()
                    )
                elif isinstance(field_value, _CONTAINER_TYPES):
                    top_child_type_counts.update(
                        _qualified_type_name(item) for item in field_value
                    )
                elif field_value is not None:
                    top_child_type_counts[_qualified_type_name(field_value)] += 1

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
        for field_name in sorted(
            {
                name
                for state in states
                for name, _value in _iter_direct_field_entries(state)
            }
        ):
            field_values = [
                field_value
                for state in states
                for name, field_value in _iter_direct_field_entries(state)
                if name == field_name
            ]
            field_stats = DeepSizeStats(max_objects=max_objects)
            field_recursive_bytes[field_name] = _exclusive_deep_size(
                field_values,
                seen=set(),
                max_depth=max_depth,
                stats=field_stats,
            )

        histograms.append(
            {
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
                "field_shallow_bytes": dict(
                    _ordered_counter_items(field_shallow_bytes)
                ),
                "field_recursive_bytes": field_recursive_bytes,
                "top_child_object_types": dict(top_child_type_counts.most_common(10)),
            }
        )

    return tuple(histograms)


def gc_shallow_size_summary(*, top_n: int) -> dict[str, object]:
    """Return one cheap process-wide shallow memory summary from GC objects."""
    type_counts = Counter[str]()
    type_bytes = Counter[str]()
    project_type_counts = Counter[str]()
    project_type_bytes = Counter[str]()
    tracked_counts = Counter[str]()
    tracked_bytes = Counter[str]()
    frozenset_ownership = _FrozensetOwnershipAccumulator()
    morpion_state_fields = _MorpionStateFrozensetAccumulator()

    object_count = 0
    total_shallow_bytes = 0
    gc_objects = gc.get_objects()
    for value in gc_objects:
        object_count += 1
        byte_count = _size_or_zero(value)
        total_shallow_bytes += byte_count
        type_name = _qualified_type_name(value)
        type_counts[type_name] += 1
        type_bytes[type_name] += byte_count
        if isinstance(value, frozenset):
            _observe_frozenset(
                frozenset_ownership,
                cast("frozenset[object]", value),
                sample_cap=_DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP,
            )
        if _is_morpion_state_type_name(type_name):
            _observe_morpion_state_frozenset_fields(morpion_state_fields, value)
        if type_name.startswith(_PROJECT_TYPE_PREFIXES):
            project_type_counts[type_name] += 1
            project_type_bytes[type_name] += byte_count
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES:
            if type_name == suffix or type_name.endswith(f".{suffix}"):
                tracked_counts[suffix] += 1
                tracked_bytes[suffix] += byte_count
                break

    top_by_bytes = type_bytes.most_common(top_n)
    top_by_count = type_counts.most_common(top_n)
    top_project_by_bytes = project_type_bytes.most_common(top_n)
    top_project_by_count = project_type_counts.most_common(top_n)
    tracked_by_bytes = [
        (suffix, tracked_bytes[suffix])
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES
        if tracked_counts[suffix] or tracked_bytes[suffix]
    ]
    tracked_by_count = [
        (suffix, tracked_counts[suffix])
        for suffix in _TRACKED_SHALLOW_TYPE_SUFFIXES
        if tracked_counts[suffix] or tracked_bytes[suffix]
    ]
    frozenset_ownership_summary = _direct_frozenset_ownership_summary(
        frozenset_ownership,
        morpion_state_fields,
    )

    return {
        "object_count": object_count,
        "total_shallow_bytes": total_shallow_bytes,
        "top_by_bytes": top_by_bytes,
        "top_by_count": top_by_count,
        "top_project_by_bytes": top_project_by_bytes,
        "top_project_by_count": top_project_by_count,
        "tracked_by_bytes": tracked_by_bytes,
        "tracked_by_count": tracked_by_count,
        "frozenset_ownership": frozenset_ownership_summary,
    }


def _log_gc_shallow_size_summary(*, event: str, top_n: int) -> None:
    start_time = time.perf_counter()
    LOGGER.info(
        "[growth-recursive-profile] event=%s gc_shallow_size_summary_start",
        event,
    )
    summary = gc_shallow_size_summary(top_n=top_n)
    LOGGER.info(
        "[growth-recursive-profile] event=%s gc_shallow_size_summary_done "
        "elapsed_s=%s object_count=%s",
        event,
        format_metric(time.perf_counter() - start_time),
        summary["object_count"],
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s histogram=gc_shallow_sizes "
        "object_count=%s total_shallow_bytes=%s total_shallow_mb=%s "
        "top_by_bytes=%s top_by_count=%s top_project_by_bytes=%s "
        "top_project_by_count=%s tracked_by_bytes=%s tracked_by_count=%s",
        event,
        summary["object_count"],
        summary["total_shallow_bytes"],
        format_metric(_mb(cast("int", summary["total_shallow_bytes"]))),
        _format_name_int_pairs(cast("list[tuple[str, int]]", summary["top_by_bytes"])),
        _format_name_int_pairs(cast("list[tuple[str, int]]", summary["top_by_count"])),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["top_project_by_bytes"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["top_project_by_count"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["tracked_by_bytes"])
        ),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", summary["tracked_by_count"])
        ),
    )
    frozenset_ownership = cast(
        "dict[str, object]",
        summary["frozenset_ownership"],
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s histogram=frozenset_ownership "
        "total_count=%s total_shallow_mb=%s len_buckets=%s "
        "morpion_state_count=%s morpion_state_field_refs=%s "
        "morpion_state_field_shallow_mb=%s "
        "morpion_state_field_len_buckets=%s",
        event,
        frozenset_ownership["total_count"],
        format_metric(_mb(cast("int", frozenset_ownership["total_shallow_bytes"]))),
        _format_name_int_pairs(
            cast("list[tuple[str, int]]", frozenset_ownership["len_buckets"])
        ),
        frozenset_ownership["morpion_state_count"],
        _format_name_int_pairs(
            cast(
                "list[tuple[str, int]]",
                frozenset_ownership["morpion_state_field_refs"],
            )
        ),
        format_metric(
            _mb(cast("int", frozenset_ownership["morpion_state_field_shallow_bytes"]))
        ),
        _format_name_int_pairs(
            cast(
                "list[tuple[str, int]]",
                frozenset_ownership["morpion_state_field_len_buckets"],
            )
        ),
    )


def linoo_state_histograms(
    selector: object | None,
    *,
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
) -> dict[str, object]:
    """Return sparse Linoo state-table diagnostics when a Linoo selector is present."""
    if selector is None:
        return {"present": False}
    linoo_selector, node_state_by_id = _resolve_linoo_selector(selector)
    if linoo_selector is None:
        raw_node_state_by_id = _raw_getattr(selector, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
        return {
            "present": True,
            "selector_type": _qualified_type_name(selector),
            "node_state_table_type": _qualified_type_name(raw_node_state_by_id),
        }
    assert isinstance(node_state_by_id, Mapping)

    default_count = 0
    non_default_count = 0
    state_type_counts = Counter[str]()
    slot_value_type_counts = Counter[str]()
    node_state_table_shallow_bytes = _size_or_zero(node_state_by_id)
    container_shallow_total = node_state_table_shallow_bytes
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
            except Exception:  # pylint: disable=broad-exception-caught
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
                    max_depth=max_depth,
                )

    return {
        "present": True,
        "selector_type": _qualified_type_name(linoo_selector),
        "node_state_count": len(node_state_by_id),
        "default_count": default_count,
        "non_default_count": non_default_count,
        "node_state_table_type": _qualified_type_name(node_state_by_id),
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
    linoo_selector, _node_state_by_id = _resolve_linoo_selector(selector)
    if linoo_selector is None:
        return ({"present": False},)

    breakdowns: list[dict[str, object]] = []
    for field_name, field_value in _iter_direct_field_entries(linoo_selector):
        recursive_reachable_bytes, stats = _measure_standalone_reachable(
            field_value,
            max_depth=max_depth,
            max_objects=max_objects,
        )
        breakdowns.append(
            {
                "present": True,
                "selector_type": _qualified_type_name(linoo_selector),
                "field_name": field_name,
                "value_type": _qualified_type_name(field_value),
                "shallow_bytes": _size_or_zero(field_value),
                "recursive_reachable_bytes": recursive_reachable_bytes,
                "visited_objects": stats.visited_objects,
                "capped": stats.capped,
                "max_depth_reached_count": stats.max_depth_reached_count,
                "recursion_error_count": stats.recursion_error_count,
            }
        )
    if not breakdowns:
        return (
            {
                "present": True,
                "selector_type": _qualified_type_name(linoo_selector),
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
    versions = _raw_getattr(linoo_selector, "_candidate_versions_by_node_id")
    present = _raw_getattr(linoo_selector, "_candidate_heap_present_by_node_id")
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
    linoo_selector, _node_state_by_id = _resolve_linoo_selector(selector)
    if linoo_selector is None:
        return {"present": False}

    candidates_by_depth = _raw_getattr(linoo_selector, "_candidates_by_depth")
    if not isinstance(candidates_by_depth, Mapping):
        return {
            "present": True,
            "selector_type": _qualified_type_name(linoo_selector),
            "candidate_heap_table_type": _qualified_type_name(candidates_by_depth),
            **_tree_reachability_flags(candidates_by_depth),
        }

    heap_type_counts = Counter[str]()
    entry_type_counts = Counter[str]()
    entry_shape_counts = Counter[str]()
    entry_value_type_counts = Counter[str]()
    heap_count = 0
    candidate_entry_count = 0
    heap_shallow_bytes = _size_or_zero(candidates_by_depth)

    for heap in candidates_by_depth.values():
        heap_count += 1
        heap_type_counts[_qualified_type_name(heap)] += 1
        heap_shallow_bytes += _size_or_zero(heap)
        if not isinstance(heap, Iterable) or isinstance(heap, str | bytes | bytearray):
            continue
        for entry in heap:
            candidate_entry_count += 1
            entry_type_counts[_qualified_type_name(entry)] += 1
            if isinstance(entry, tuple):
                entry_shape_counts[f"tuple[{len(entry)}]"] += 1
                for item in entry:
                    entry_value_type_counts[_qualified_type_name(item)] += 1
            else:
                entry_shape_counts[_qualified_type_name(entry)] += 1

    recursive_bytes, stats = _measure_standalone_reachable(
        candidates_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    stale_count = _linoo_candidate_stale_count(linoo_selector, candidates_by_depth)

    return {
        "present": True,
        "selector_type": _qualified_type_name(linoo_selector),
        "candidate_depth_count": len(candidates_by_depth),
        "candidate_heap_count": heap_count,
        "candidate_entry_count": candidate_entry_count,
        "candidate_stale_entry_count": stale_count,
        "candidate_stale_fraction": (
            None
            if stale_count is None or candidate_entry_count == 0
            else format_metric(stale_count / candidate_entry_count)
        ),
        "candidate_heap_table_shallow_bytes": _size_or_zero(candidates_by_depth),
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
    linoo_selector, node_state_by_id = _resolve_linoo_selector(selector)
    if linoo_selector is None:
        return {"present": False}

    selector_recursive_bytes, selector_stats = _measure_standalone_reachable(
        linoo_selector,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    node_state_table = node_state_by_id if node_state_by_id is not None else {}
    table_recursive_bytes, table_stats = _measure_standalone_reachable(
        node_state_table,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    node_state_object_shallow_total = 0
    status_breakdown = Counter[str]()
    for state in node_state_table.values():
        node_state_object_shallow_total += _size_or_zero(state)
        status = _raw_getattr(state, "status")
        status_breakdown[str(status)] += 1

    candidates_by_depth = _raw_getattr(linoo_selector, "_candidates_by_depth")
    candidate_heap = linoo_candidate_heap_histogram(
        linoo_selector,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    depth_stats_by_depth = _raw_getattr(linoo_selector, "_depth_stats_by_depth")
    depth_stats_recursive_bytes, depth_stats_recursive = _measure_standalone_reachable(
        depth_stats_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )
    frontier_ids_by_depth = _raw_getattr(linoo_selector, "_frontier_node_ids_by_depth")
    frontier_recursive_bytes, frontier_stats = _measure_standalone_reachable(
        frontier_ids_by_depth,
        max_depth=max_depth,
        max_objects=max_objects,
    )

    return {
        "present": True,
        "selector_type": _qualified_type_name(linoo_selector),
        "total_selector_recursive_bytes": selector_recursive_bytes,
        "total_selector_recursive_visited_objects": selector_stats.visited_objects,
        "total_selector_recursive_capped": selector_stats.capped,
        "node_state_table_attr_name": _LINOO_NODE_STATE_TABLE_ATTR_NAME,
        "node_state_table_type": _qualified_type_name(node_state_table),
        "node_state_table_shallow_bytes": _size_or_zero(node_state_table),
        "node_state_table_recursive_bytes": table_recursive_bytes,
        "node_state_table_recursive_visited_objects": table_stats.visited_objects,
        "node_state_table_recursive_capped": table_stats.capped,
        "node_state_count": len(node_state_table),
        "node_state_object_shallow_total_bytes": node_state_object_shallow_total,
        "status_representation_breakdown": dict(
            _ordered_counter_items(status_breakdown)
        ),
        "candidates_by_depth_type": _qualified_type_name(candidates_by_depth),
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
        "depth_stats_type": _qualified_type_name(depth_stats_by_depth),
        "depth_stats_count": (
            len(depth_stats_by_depth)
            if isinstance(depth_stats_by_depth, Sized)
            else None
        ),
        "depth_stats_shallow_bytes": _size_or_zero(depth_stats_by_depth),
        "depth_stats_recursive_bytes": depth_stats_recursive_bytes,
        "depth_stats_recursive_visited_objects": depth_stats_recursive.visited_objects,
        "depth_stats_recursive_capped": depth_stats_recursive.capped,
        "frontier_ids_by_depth_type": _qualified_type_name(frontier_ids_by_depth),
        "frontier_ids_by_depth_shallow_bytes": _size_or_zero(frontier_ids_by_depth),
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
    linoo_selector, node_state_by_id = _resolve_linoo_selector(selector)
    if linoo_selector is None or node_state_by_id is None:
        return {"present": False}

    key_type_counts = Counter[str]()
    value_type_counts = Counter[str]()
    node_states_shallow_bytes = 0
    node_state_count = 0
    for key, value in node_state_by_id.items():
        key_type_counts[_qualified_type_name(key)] += 1
        value_type_counts[_qualified_type_name(value)] += 1
        node_states_shallow_bytes += _size_or_zero(value)
        node_state_count += 1

    table_recursive_reachable_bytes, table_stats = _measure_standalone_reachable(
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
        "selector_type": _qualified_type_name(linoo_selector),
        "table_attr_name": _LINOO_NODE_STATE_TABLE_ATTR_NAME,
        "table_type": _qualified_type_name(node_state_by_id),
        "table_length": len(node_state_by_id),
        "table_shallow_bytes": _size_or_zero(node_state_by_id),
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
    linoo_selector, node_state_by_id = _resolve_linoo_selector(selector)
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
            "selector_type": _qualified_type_name(linoo_selector),
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
            present, slot_value = _raw_getattr_present(state, slot_name)
            if not present:
                continue
            slot_value_type_counts[_qualified_type_name(slot_value)] += 1
            slot_value_kind_counts[_linoo_slot_value_kind(slot_value)] += 1
            slot_observation_counts[slot_name] += 1
            slot_shallow_bytes[slot_name] += _size_or_zero(slot_value)
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
        "selector_type": _qualified_type_name(linoo_selector),
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


def _handle_node_id(handle: object) -> object | None:
    """Return the raw checkpoint node id from known handle layouts."""
    for attr_name in ("node_id", "_node_id"):
        value = _raw_getattr(handle, attr_name)
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
        value = _raw_getattr(resolver, attr_name)
        if isinstance(value, Mapping):
            return attr_name, value
    owner = _raw_getattr(resolver, "owner")
    if owner is None:
        return None, None
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        value = _raw_getattr(owner, attr_name)
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
        value = _raw_getattr(resolver, attr_name)
        if isinstance(value, Mapping):
            return value
    return None


def _handle_has_materialized_state(handle: object, resolver: object | None) -> bool:
    """Return whether one raw checkpoint handle already points to materialized state."""
    for attr_name in ("state_", "_state", "_materialized_state"):
        present, value = _raw_getattr_present(handle, attr_name)
        if present and value is not None:
            return True
    node_id = _handle_node_id(handle)
    if resolver is None or not isinstance(node_id, int):
        return False
    resolved_states = _resolver_resolved_states(resolver)
    return isinstance(resolved_states, Mapping) and node_id in resolved_states


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
        if _deep_size_stats_capped(payload_stats):
            break
        payload_store_count += 1
        for payload in payload_store.payloads.values():
            if _deep_size_stats_capped(payload_stats):
                break
            add_payload(payload)

    for node in nodes:
        if max_handles is not None and handles_seen >= max_handles:
            handle_scan_capped = True
            break
        handle = _tree_node_slot(node, "state_handle_")
        if handle is None:
            continue
        handles_seen += 1
        handle_type_counts[_qualified_type_name(handle)] += 1

        state_value = _raw_getattr(handle, "state_")
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
        handle = _tree_node_slot(node, "state_handle_")
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
                resolver_type=_qualified_type_name(resolver),
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
        histograms.append(
            {
                "resolver_type": payload_store.resolver_type,
                "resolver_object_id": payload_store.resolver_id,
                "payload_mapping_attr_name": payload_store.attr_name,
                "payload_mapping_type": _qualified_type_name(payload_store.payloads),
                "payload_mapping_length": len(payload_store.payloads),
                "anchor_count": payload_store.anchor_count,
                "delta_count": payload_store.delta_count,
                "payload_mapping_recursive_bytes": (
                    0 if record is None else record.bytes
                ),
                "payload_mapping_shallow_bytes": _size_or_zero(payload_store.payloads),
                "checkpoint_backed_state_handle_count": (
                    0
                    if resolver_stats is None
                    else resolver_stats.checkpoint_handle_count
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
            }
        )

    for resolver_id, resolver_stats in handle_stats_by_resolver_id.items():
        if resolver_id in seen_store_resolver_ids:
            continue
        histograms.append(
            {
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
            }
        )

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
        return _qualified_type_name(value)
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
    return _qualified_type_name(value)


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
        counts[_qualified_type_name(value)] += 1
        if isinstance(value, Mapping):
            for key, item in value.items():
                counts[_qualified_type_name(key)] += 1
                counts[_qualified_type_name(item)] += 1
        elif isinstance(value, list | tuple):
            for item in value[:_DEFAULT_PAYLOAD_SHAPE_SAMPLE_ITEMS]:
                counts[_qualified_type_name(item)] += 1
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
            _raw_getattr(payload, "anchor_ref") for payload in anchor_payloads
        ]
        delta_refs = [_raw_getattr(payload, "delta_ref") for payload in delta_payloads]
        state_summaries = [
            _raw_getattr(payload, "state_summary") for payload in payloads
        ]
        state_parent_branches = [
            _raw_getattr(payload, "state_parent_branch") for payload in delta_payloads
        ]
        state_parent_node_ids = [
            _raw_getattr(payload, "state_parent_node_id") for payload in delta_payloads
        ]
        payload_type_counts = Counter[str](
            _qualified_type_name(payload) for payload in payloads
        )

        histograms.append(
            {
                "resolver_type": payload_store.resolver_type,
                "resolver_object_id": payload_store.resolver_id,
                "payload_store_type": _qualified_type_name(payload_store.payloads),
                "payload_store_length": len(payload_store.payloads),
                "total_payload_count": len(payloads),
                "anchor_payload_count": len(anchor_payloads),
                "delta_payload_count": len(delta_payloads),
                "payload_type_counts": dict(
                    _ordered_counter_items(payload_type_counts)
                ),
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
                "anchor_ref_common_dict_keys": _payload_shape_dict_key_counts(
                    anchor_refs
                ),
                "delta_ref_common_dict_keys": _payload_shape_dict_key_counts(
                    delta_refs
                ),
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
                    _small_payload_sample(state_summaries[0])
                    if state_summaries
                    else None
                ),
            }
        )
    if histograms:
        return tuple(histograms)
    return ({"present": False},)


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
        store_type_counts[_qualified_type_name(store)] += 1
        for payload in store.values():
            payload_count += 1
            payload_type_counts[_qualified_type_name(payload)] += 1
            payload_kind = _checkpoint_payload_kind(payload)
            state_summary = _raw_getattr(payload, "state_summary")
            if state_summary is not None:
                state_summaries.append(state_summary)
            if payload_kind == "anchor":
                anchor_payloads.append(payload)
                anchor_ref = _raw_getattr(payload, "anchor_ref")
                if anchor_ref is not None:
                    anchor_refs.append(anchor_ref)
            elif payload_kind == "delta":
                delta_payloads.append(payload)
                delta_ref = _raw_getattr(payload, "delta_ref")
                if delta_ref is not None:
                    delta_refs.append(delta_ref)
                state_parent_node_id = _raw_getattr(payload, "state_parent_node_id")
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


def log_growth_recursive_memory_profile(
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


__all__ = [
    "DeepSizeStats",
    "_find_linoo_selector_root",
    "build_recursive_profile_context",
    "checkpoint_payload_lifetime_histograms",
    "checkpoint_payload_shape_histograms",
    "checkpoint_state_histograms",
    "checkpoint_state_roots_detail_histogram",
    "child_link_storage_detail_histogram",
    "deep_size",
    "frozenset_ownership_histogram",
    "linoo_candidate_heap_histogram",
    "linoo_selector_detail_histogram",
    "linoo_state_histograms",
    "log_growth_recursive_memory_profile",
    "node_evaluation_runtime_detail_histograms",
    "node_evaluation_runtime_histograms",
    "parent_link_storage_histogram",
    "slot_names",
    "state_eviction_runtime_histogram",
    "state_handle_materialization_detail_histogram",
    "state_retention_by_node_status_histogram",
    "tree_topology_histograms",
]
