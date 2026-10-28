"""Cached Morpion training index schedules."""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .streaming import (
    is_streaming_validation_index,
    streaming_row_is_in_split,
    streaming_split_policy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class CachedIndexSchedule:
    """Train/validation row-index schedule for one cached dataset."""

    row_count: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    split_policy: str


def cached_index_schedule(
    *,
    row_count: int,
    validation_fraction: float,
    validation_indices: frozenset[int] | None = None,
) -> CachedIndexSchedule:
    """Build deterministic train/validation indices matching streaming policy."""
    train_indices: list[int] = []
    selected_validation_indices: list[int] = []
    for row_index in range(row_count):
        if is_streaming_validation_index(
            row_index,
            validation_fraction=validation_fraction,
            validation_indices=validation_indices,
        ):
            selected_validation_indices.append(row_index)
        else:
            train_indices.append(row_index)
    return CachedIndexSchedule(
        row_count=row_count,
        train_indices=tuple(train_indices),
        validation_indices=tuple(selected_validation_indices),
        split_policy="canonical_state_sha256_v1"
        if validation_indices is not None
        else streaming_split_policy(validation_fraction),
    )


def shuffled_epoch_train_indices(
    *,
    train_indices: tuple[int, ...],
    shuffle: bool,
    validation_seed: int,
    epoch_index: int,
) -> tuple[int, ...]:
    """Return epoch train indices, globally shuffled when requested."""
    if not shuffle:
        return train_indices
    shuffled = list(train_indices)
    rng = random.Random(validation_seed + epoch_index)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def index_batches(
    indices: tuple[int, ...],
    *,
    batch_size: int,
) -> Iterable[tuple[int, ...]]:
    """Yield fixed-size index batches."""
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if batch:
            yield batch


def split_indices_for_streaming_policy(
    *,
    row_count: int,
    validation_fraction: float,
    validation_indices: frozenset[int] | None = None,
    split: Literal["train", "validation"],
) -> tuple[int, ...]:
    """Return train or validation indices for the streaming split policy."""
    return tuple(
        row_index
        for row_index in range(row_count)
        if streaming_row_is_in_split(
            row_index,
            validation_fraction=validation_fraction,
            validation_indices=validation_indices,
            split=split,
        )
    )
