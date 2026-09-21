"""Streaming Morpion neural-network training helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from chipiron.learning.supervised import train_regression_batch
from chipiron.learning.timing import PhaseDurations

from .row_batches import indexed_row_chunks, row_batches, rows_to_sample_batch

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from chipiron.environments.morpion.learning import MorpionSupervisedRow
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .args import MorpionTrainingArgs


@dataclass(frozen=True, slots=True)
class StreamingEpochStats:
    """Stats returned after one streaming training epoch."""

    chunk_count: int
    train_count: int
    validation_count: int
    batch_count: int
    loss: float
    phase_durations: dict[str, float]
    elapsed_seconds: float


def train_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> StreamingEpochStats:
    """Train one streaming epoch over persisted Morpion row chunks."""
    epoch_started_at = perf_counter()
    epoch_timings = PhaseDurations()
    model.train()
    chunk_count = 0
    train_count = 0
    validation_count = 0
    batch_count = 0
    squared_error_sum = 0.0
    value_count = 0
    rng = random.Random(args.validation_seed + epoch_index)
    row_chunk_iter = iter(
        indexed_row_chunks(
            args.dataset_file,
            chunk_size=row_chunk_size,
            max_rows=max_rows,
        )
    )
    while True:
        try:
            with epoch_timings.time_phase("chunk_load"):
                row_start_index, rows = next(row_chunk_iter)
        except StopIteration:
            break
        chunk_count += 1
        with epoch_timings.time_phase("split_select"):
            train_rows: list[MorpionSupervisedRow] = []
            for offset, row in enumerate(rows):
                row_index = row_start_index + offset
                if is_streaming_validation_index(
                    row_index,
                    validation_fraction=args.validation_fraction,
                ):
                    validation_count += 1
                else:
                    train_rows.append(row)
            if args.shuffle:
                rng.shuffle(train_rows)
        train_count += len(train_rows)
        row_batch_iter = iter(row_batches(train_rows, batch_size=args.batch_size))
        while True:
            try:
                with epoch_timings.time_phase("row_batching"):
                    row_batch = next(row_batch_iter)
            except StopIteration:
                break
            batch_count += 1
            with epoch_timings.time_phase("row_to_sample_batch"):
                sample_batch = rows_to_sample_batch(row_batch, args=args)
            batch_stats = train_regression_batch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                batch=sample_batch,
                device=device,
                timings=epoch_timings,
            )
            squared_error_sum += batch_stats.squared_error_sum
            value_count += batch_stats.target_count
        if progress_callback is not None:
            with epoch_timings.time_phase("progress_callback"):
                progress_callback(
                    chunk_count,
                    row_start_index + len(rows),
                    epoch_index + 1,
                    args.num_epochs,
                )
    elapsed_seconds = perf_counter() - epoch_started_at
    epoch_timings.add_duration("epoch_total", elapsed_seconds)
    loss_value = 0.0 if value_count == 0 else squared_error_sum / value_count
    return StreamingEpochStats(
        chunk_count=chunk_count,
        train_count=train_count,
        validation_count=validation_count,
        batch_count=batch_count,
        loss=loss_value,
        phase_durations=epoch_timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def streaming_row_is_in_split(
    row_index: int,
    *,
    validation_fraction: float,
    split: Literal["train", "validation"],
) -> bool:
    """Return whether one row index belongs to a streaming split."""
    is_validation = is_streaming_validation_index(
        row_index,
        validation_fraction=validation_fraction,
    )
    return is_validation if split == "validation" else not is_validation


def is_streaming_validation_index(
    row_index: int,
    *,
    validation_fraction: float,
) -> bool:
    """Return whether one row index belongs to streaming validation."""
    if validation_fraction <= 0.0:
        return False
    period = max(2, round(1.0 / validation_fraction))
    return (row_index + 1) % period == 0


def streaming_split_policy(validation_fraction: float) -> str:
    """Return the manifest/model metadata name for the streaming validation split."""
    if validation_fraction <= 0.0:
        return "none"
    return f"index_modulo_{max(2, round(1.0 / validation_fraction))}"


def morpion_streaming_split_policy(validation_fraction: float) -> str:
    """Return the manifest/model metadata name for the streaming validation split."""
    return streaming_split_policy(validation_fraction)
