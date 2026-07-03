"""Cached Morpion training index schedules."""

from __future__ import annotations

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
) -> CachedIndexSchedule:
    """Build deterministic train/validation indices matching streaming policy."""
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for row_index in range(row_count):
        if is_streaming_validation_index(
            row_index,
            validation_fraction=validation_fraction,
        ):
            validation_indices.append(row_index)
        else:
            train_indices.append(row_index)
    return CachedIndexSchedule(
        row_count=row_count,
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        split_policy=streaming_split_policy(validation_fraction),
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
    split: Literal["train", "validation"],
) -> tuple[int, ...]:
    """Return train or validation indices for the streaming split policy."""
    return tuple(
        row_index
        for row_index in range(row_count)
        if streaming_row_is_in_split(
            row_index,
            validation_fraction=validation_fraction,
            split=split,
        )
    )
