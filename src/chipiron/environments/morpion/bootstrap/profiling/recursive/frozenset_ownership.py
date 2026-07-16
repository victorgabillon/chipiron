"""Frozenset ownership diagnostics for Morpion recursive memory profiling."""

from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field
from types import FrameType
from typing import TYPE_CHECKING, cast

from .deep_size import size_or_zero
from .object_access import (
    qualified_type_name,
    raw_getattr,
    safe_object_dict,
    should_skip_deep,
    slot_names,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP = 5_000
_DEFAULT_FROZENSET_OWNER_REFERRER_SCAN_CAP = 16
_FROZENSET_MORPION_STATE_FIELDS = (
    "points",
    "used_unit_segments",
    "played_moves",
)

__all__ = [
    "direct_frozenset_ownership_summary",
    "frozenset_ownership_histogram",
]


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


def direct_frozenset_ownership_summary(
    objects: Iterable[object],
    *,
    sample_cap: int = _DEFAULT_FROZENSET_OWNERSHIP_SAMPLE_CAP,
) -> dict[str, object]:
    """Return direct frozenset and MorpionState frozenset-field counts."""
    frozenset_ownership = _FrozensetOwnershipAccumulator()
    morpion_state_fields = _MorpionStateFrozensetAccumulator()
    effective_sample_cap = max(0, sample_cap)

    for value in objects:
        if isinstance(value, frozenset):
            _observe_frozenset(
                frozenset_ownership,
                cast("frozenset[object]", value),
                sample_cap=effective_sample_cap,
            )
        if _is_morpion_state_type_name(qualified_type_name(value)):
            _observe_morpion_state_frozenset_fields(morpion_state_fields, value)

    return _direct_frozenset_ownership_summary(
        frozenset_ownership,
        morpion_state_fields,
    )


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


def _observe_frozenset(
    accumulator: _FrozensetOwnershipAccumulator,
    value: frozenset[object],
    *,
    sample_cap: int,
) -> None:
    accumulator.total_count += 1
    accumulator.total_shallow_bytes += size_or_zero(value)
    accumulator.len_bucket_counts[_frozenset_len_bucket(len(value))] += 1
    if len(accumulator.sampled_frozensets) < sample_cap:
        accumulator.sampled_frozensets.append(value)


def _observe_morpion_state_frozenset_fields(
    accumulator: _MorpionStateFrozensetAccumulator,
    state: object,
) -> None:
    accumulator.morpion_state_count += 1
    for field_name in _FROZENSET_MORPION_STATE_FIELDS:
        field_value = raw_getattr(state, field_name)
        if not isinstance(field_value, frozenset):
            continue
        accumulator.field_ref_counts[field_name] += 1
        accumulator.field_len_bucket_counts[
            _frozenset_len_bucket(len(field_value))
        ] += 1
        accumulator.total_field_shallow_bytes += size_or_zero(field_value)


def _is_morpion_state_type_name(type_name: str) -> bool:
    return type_name == "MorpionState" or type_name.endswith(".MorpionState")


def _morpion_state_field_ref(
    referrer: object,
    target: frozenset[object],
) -> str | None:
    owner_dict = safe_object_dict(referrer)
    if owner_dict is not None:
        for field_name in _FROZENSET_MORPION_STATE_FIELDS:
            if owner_dict.get(field_name) is target:
                return field_name
    for slot_name in slot_names(referrer):
        if slot_name not in _FROZENSET_MORPION_STATE_FIELDS:
            continue
        if raw_getattr(referrer, slot_name) is target:
            return slot_name
    return None


def _is_skippable_owner_referrer(value: object) -> bool:
    return isinstance(value, FrameType) or should_skip_deep(value)


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
        owner_type_name = qualified_type_name(owner)
        if not _is_morpion_state_type_name(owner_type_name):
            continue
        owner_dict = safe_object_dict(owner)
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
    referrer_type_name = qualified_type_name(referrer)
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
    base_ignored_referrer_ids.update({
        id(accumulator),
        id(accumulator.len_bucket_counts),
        id(accumulator.sampled_frozensets),
        id(base_ignored_referrer_ids),
    })

    for frozen_set in accumulator.sampled_frozensets:
        iteration_ignored_referrer_ids = set(base_ignored_referrer_ids)
        seen_owner_field_refs: set[tuple[int, str]] = set()
        for referrer in gc.get_referrers(frozen_set):
            if id(referrer) in iteration_ignored_referrer_ids:
                continue
            if _is_skippable_owner_referrer(referrer):
                continue
            referrer_type_name = qualified_type_name(referrer)
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
