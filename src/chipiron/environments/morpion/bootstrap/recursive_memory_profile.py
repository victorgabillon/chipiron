"""Recursive restored-tree memory diagnostics for Morpion growth runtimes."""

from __future__ import annotations

import gc
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    FrameType,
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

_DEFAULT_CHECKPOINT_HANDLE_SCAN_CAP = 50_000
_DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP = 100
_DEFAULT_RECURSIVE_PROFILE_MAX_OBJECTS = 3_000_000
_DEFAULT_DEEP_SIZE_MAX_DEPTH = 64
_DEFAULT_LINOO_NODE_STATE_SAMPLE_CAP = 5_000
_DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP = 5_000
_DEFAULT_FROZENSET_OWNER_REFERRER_SCAN_CAP = 16
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
    "int",
    "str",
    "enum",
    "tuple",
    "list",
    "dict",
    "set",
    "None",
)
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
class DeepSizeStats:
    """Mutable counters for one recursive-size traversal."""

    visited_objects: int = 0
    max_objects: int | None = None
    capped: bool = False
    max_depth_reached_count: int = 0
    recursion_error_count: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointPayloadStore:
    """Concrete mapping that owns checkpoint state payload objects."""

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
    return "[" + ",".join(f"{name}:{format_metric(value)}" for name, value in items) + "]"


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


def _should_skip_deep(value: object) -> bool:
    return isinstance(value, _SKIP_DEEP_TYPES)


def _raw_getattr(value: object, attr_name: str) -> object | None:
    """Read a concrete attribute/slot without using ``dir`` or properties."""
    try:
        result: object = object.__getattribute__(value, attr_name)
    except Exception:
        return None
    return result


def _raw_getattr_present(value: object, attr_name: str) -> tuple[bool, object | None]:
    """Return whether a concrete attribute exists plus its raw value."""
    try:
        result: object = object.__getattribute__(value, attr_name)
    except Exception:
        return False, None
    return True, result


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


def _iter_direct_field_entries(value: object) -> Iterator[tuple[str, object]]:
    """Yield raw direct fields from __dict__ entries and declared slots."""
    seen_names: set[str] = set()
    raw_dict = _safe_object_dict(value)
    if raw_dict is not None:
        for field_name, field_value in raw_dict.items():
            if not isinstance(field_name, str) or field_name in seen_names:
                continue
            seen_names.add(field_name)
            yield field_name, field_value
    for slot_name in slot_names(value):
        if slot_name in seen_names:
            continue
        present, slot_value = _raw_getattr_present(value, slot_name)
        if not present:
            continue
        seen_names.add(slot_name)
        yield slot_name, slot_value


def _measure_standalone_reachable(
    value: object,
    *,
    max_depth: int | None,
    max_objects: int | None,
) -> tuple[int, DeepSizeStats]:
    stats = DeepSizeStats(max_objects=max_objects)
    byte_count = deep_size(value, seen=set(), max_depth=max_depth, stats=stats)
    return byte_count, stats


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
    if _qualified_type_name(value).endswith("AlgorithmNode"):
        return "AlgorithmNode"
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
    return _qualified_type_name(value)


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
        accumulator.field_len_bucket_counts[_frozenset_len_bucket(len(field_value))] += 1
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
    max_depth: int | None = _DEFAULT_DEEP_SIZE_MAX_DEPTH,
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


def _mark_recursion_error(stats: DeepSizeStats) -> None:
    stats.recursion_error_count += 1
    stats.capped = True


def _try_push_deep_size_object(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None,
    depth: int,
    stack: list[tuple[object, int]],
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
        stats.max_depth_reached_count += 1
        stats.capped = True
        return size

    stack.append((obj, depth))
    return size


def _deep_size(
    obj: object,
    *,
    seen: set[int],
    max_depth: int | None,
    depth: int,
    stats: DeepSizeStats,
) -> int:
    total_size = _try_push_deep_size_object(
        obj,
        seen=seen,
        max_depth=max_depth,
        depth=depth,
        stack=(stack := []),
        stats=stats,
    )

    while stack:
        current, current_depth = stack.pop()
        next_depth = current_depth + 1

        if isinstance(current, Mapping):
            try:
                for key, value in current.items():
                    total_size += _try_push_deep_size_object(
                        key,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
                    total_size += _try_push_deep_size_object(
                        value,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
            except RecursionError:
                _mark_recursion_error(stats)
            continue

        if isinstance(current, _CONTAINER_TYPES):
            try:
                for item in current:
                    total_size += _try_push_deep_size_object(
                        item,
                        seen=seen,
                        max_depth=max_depth,
                        depth=next_depth,
                        stack=stack,
                        stats=stats,
                    )
            except RecursionError:
                _mark_recursion_error(stats)
            continue

        try:
            for attr_value in _iter_object_attribute_values(current):
                total_size += _try_push_deep_size_object(
                    attr_value,
                    seen=seen,
                    max_depth=max_depth,
                    depth=next_depth,
                    stack=stack,
                    stats=stats,
                )
        except RecursionError:
            _mark_recursion_error(stats)

    return total_size


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
    if isinstance(value, _CONTAINER_TYPES):
        yield from value
        return

    raw_dict = _safe_object_dict(value)
    if raw_dict is not None:
        yield from raw_dict.values()
    for slot_name in slot_names(value):
        slot_value = _raw_getattr(value, slot_name)
        if slot_value is not None:
            yield slot_value


def _find_linoo_selector_root(root: object | None) -> object | None:
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

        if isinstance(value, _ATOMIC_TYPES) or _should_skip_deep(value):
            continue

        node_state_by_id = _raw_getattr(value, "_node_state_by_id")
        if isinstance(node_state_by_id, Mapping):
            return value

        stack.extend(_iter_linoo_selector_search_children(value))
    return None


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


def _iter_named_raw_attribute_values(value: object) -> Iterator[tuple[str, object]]:
    raw_dict = _safe_object_dict(value)
    if raw_dict is not None:
        for attr_name, attr_value in raw_dict.items():
            if isinstance(attr_name, str):
                yield attr_name, attr_value
    for slot_name in slot_names(value):
        slot_value = _raw_getattr(value, slot_name)
        if slot_value is not None:
            yield slot_name, slot_value


def _find_checkpoint_payload_stores(
    roots: Iterable[object | None],
) -> tuple[CheckpointPayloadStore, ...]:
    """Find checkpoint payload-owner mappings without consulting properties."""
    stores: list[CheckpointPayloadStore] = []
    seen: set[int] = set()
    checked_mapping_ids: set[int] = set()
    payload_mapping_ids: set[int] = set()
    stack = [root for root in roots if root is not None]

    while stack:
        value = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)

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

        for attr_name, attr_value in _iter_named_raw_attribute_values(value):
            if isinstance(attr_value, Mapping):
                mapping_id = id(attr_value)
                if mapping_id not in checked_mapping_ids:
                    checked_mapping_ids.add(mapping_id)
                    anchor_count, delta_count = _checkpoint_payload_counts(attr_value)
                    if anchor_count or delta_count:
                        payload_mapping_ids.add(mapping_id)
                        stores.append(
                            CheckpointPayloadStore(
                                owner_type=_qualified_type_name(value),
                                attr_name=attr_name,
                                payloads=attr_value,
                                anchor_count=anchor_count,
                                delta_count=delta_count,
                            )
                        )
                        continue
                elif mapping_id in payload_mapping_ids:
                    continue
            stack.append(attr_value)

    return tuple(stores)


def _append_checkpoint_payload_store_if_payload_mapping(
    *,
    stores: list[CheckpointPayloadStore],
    checked_mapping_ids: set[int],
    payload_mapping_ids: set[int],
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
            owner_type=_qualified_type_name(owner),
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
        mapping = _raw_getattr(candidate, attr_name)
        if isinstance(mapping, Mapping):
            _append_checkpoint_payload_store_if_payload_mapping(
                stores=stores,
                checked_mapping_ids=checked_mapping_ids,
                payload_mapping_ids=payload_mapping_ids,
                owner=candidate,
                attr_name=attr_name,
                mapping=mapping,
            )
    owner = _raw_getattr(candidate, "owner")
    if owner is None:
        return
    for attr_name in _KNOWN_CHECKPOINT_STORE_ATTR_NAMES:
        mapping = _raw_getattr(owner, attr_name)
        if isinstance(mapping, Mapping):
            _append_checkpoint_payload_store_if_payload_mapping(
                stores=stores,
                checked_mapping_ids=checked_mapping_ids,
                payload_mapping_ids=payload_mapping_ids,
                owner=owner,
                attr_name=f"owner.{attr_name}",
                mapping=mapping,
            )


def _find_checkpoint_payload_stores_from_known_paths_only(
    *,
    runner: object,
    runtime: object | None,
) -> tuple[CheckpointPayloadStore, ...]:
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
        add_candidate(_raw_attr_path(runner, attr_path))
    if runtime is not None:
        add_candidate(runtime)
        for attr_path in _CHECKPOINT_ROOT_ATTR_PATHS:
            add_candidate(_raw_attr_path(runtime, attr_path))
    return tuple(stores)


def _find_checkpoint_payload_stores_from_handle_fallback(
    nodes: Sequence[object],
    *,
    handle_cap: int,
) -> tuple[CheckpointPayloadStore, ...]:
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


def _unique_roots(roots: Iterable[object | None]) -> tuple[object, ...]:
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


def _profile_nodes_from_runner(runner: object) -> tuple[object, ...]:
    return _profile_nodes_from_runner_capped(runner, node_cap=None)


def _profile_nodes_from_runner_capped(
    runner: object,
    *,
    node_cap: int | None,
) -> tuple[object, ...]:
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
            return _materialize_profile_nodes(iterator, node_cap=node_cap)

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
            return _materialize_profile_nodes(iterator, node_cap=node_cap)
    return ()


def _materialize_profile_nodes(
    values: Iterable[object],
    *,
    node_cap: int | None,
) -> tuple[object, ...]:
    if node_cap is None:
        return tuple(values)
    materialized: list[object] = []
    for value in values:
        materialized.append(value)
        if len(materialized) >= node_cap:
            break
    return tuple(materialized)


def _count_profile_branches(nodes: Iterable[object]) -> int:
    branch_count = 0
    for node in nodes:
        branch_count += sum(
            1 for _ in _iter_child_branch_refs(_tree_node_slot(node, "branches_children_"))
        )
        branch_count += sum(
            1 for _ in _iter_parent_branch_refs(_tree_node_slot(node, "parent_nodes_"))
        )
    return branch_count


def _checkpoint_store_handle_discovery_cap(
    *,
    node_cap: int | None,
    handle_cap: int = _DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP,
) -> int:
    if node_cap is None:
        return handle_cap
    return min(node_cap, handle_cap)


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
    runtime = _first_attr_path(runner, _RUNTIME_ATTR_PATHS)
    selector_root = _first_attr_path(runner, _SELECTOR_ATTR_PATHS)
    linoo_selector = _find_linoo_selector_root(selector_root)
    if linoo_selector is None and runtime is not None:
        linoo_selector = _find_linoo_selector_root(runtime)
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
    nodes = _profile_nodes_from_runner_capped(runner, node_cap=node_cap)
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
        branch_count if branch_count is not None else _count_profile_branches(nodes)
    )
    LOGGER.info(
        "[growth-recursive-profile] event=%s context_build_branches_done "
        "branch_count=%s elapsed_s=%s",
        event,
        resolved_branch_count,
        format_metric(time.perf_counter() - branches_start),
    )
    checkpoint_roots = _all_attr_paths(runner, _CHECKPOINT_ROOT_ATTR_PATHS)
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
    checkpoint_payload_stores = _find_checkpoint_payload_stores_from_known_paths_only(
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
        handle_cap = _checkpoint_store_handle_discovery_cap(node_cap=node_cap)
        LOGGER.info(
            "[growth-recursive-profile] event=%s "
            "context_build_checkpoint_stores_handle_fallback_start handle_cap=%s",
            event,
            handle_cap,
        )
        fallback_checkpoint_stores_start = time.perf_counter()
        checkpoint_payload_stores = _find_checkpoint_payload_stores_from_handle_fallback(
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
    checkpoint_roots_with_payloads = _unique_roots(
        (*checkpoint_roots, *(store.payloads for store in checkpoint_payload_stores))
    )
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
        evaluator_roots=_all_attr_paths(runner, _EVALUATOR_ATTR_PATHS),
        nodes=nodes,
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
        node_state_by_id = _raw_getattr(selector, _LINOO_NODE_STATE_TABLE_ATTR_NAME)
        return {
            "present": True,
            "selector_type": _qualified_type_name(selector),
            "node_state_table_type": _qualified_type_name(node_state_by_id),
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
            _ordered_counter_items(slot_value_kind_counts, order=_LINOO_SLOT_VALUE_KIND_ORDER)
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
    payload_stats = DeepSizeStats(max_objects=max_objects)
    resolved_recursive_seen: set[int] = set()
    resolved_recursive_bytes = 0
    resolved_stats = DeepSizeStats(max_objects=max_objects)
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
        if payload_stats.capped:
            break
        payload_store_count += 1
        for payload in payload_store.payloads.values():
            if payload_stats.capped:
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
        node_id = _raw_getattr(handle, "node_id")
        payloads = _resolver_payloads(resolver)
        if (
            not payload_stats.capped
            and isinstance(node_id, int)
            and isinstance(payloads, Mapping)
        ):
            payload = payloads.get(node_id)
            if payload is not None:
                add_payload(payload)
        resolved_states = _raw_getattr(resolver, "_resolved_states")
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
    return "[" + ",".join(record.component for record in records if record.capped is capped) + "]"


def _largest_components(records: Iterable[ComponentProfileRecord], *, limit: int) -> str:
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
        "node_evaluation_runtime",
        node_evaluation_runtime_histograms(context.nodes),
    )
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
    "checkpoint_state_histograms",
    "deep_size",
    "frozenset_ownership_histogram",
    "linoo_state_histograms",
    "log_growth_recursive_memory_profile",
    "node_evaluation_runtime_histograms",
    "slot_names",
    "tree_topology_histograms",
]
