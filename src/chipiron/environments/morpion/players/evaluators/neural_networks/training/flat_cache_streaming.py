"""Streaming training loops backed by cached Morpion flat tensors."""

from __future__ import annotations

import random
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import torch

from chipiron.learning.supervised import (
    RegressionEvaluationStats,
    evaluate_regression_batch,
    infer_batch_sample_count,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations

from .flat_tensor_cache import FlatTensorCache, flat_cache_batch
from .streaming import (
    StreamingEpochStats,
    is_streaming_validation_index,
    streaming_row_is_in_split,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .args import MorpionTrainingArgs


def train_flat_cache_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    cache: FlatTensorCache,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached Morpion flat tensor artifact."""
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
    row_limit = _cache_row_limit(cache=cache, max_rows=max_rows)
    for chunk_start in range(0, row_limit, row_chunk_size):
        with epoch_timings.time_phase("chunk_load"):
            chunk_end = min(chunk_start + row_chunk_size, row_limit)
            chunk_indices = tuple(range(chunk_start, chunk_end))
        chunk_count += 1
        with epoch_timings.time_phase("split_select"):
            train_indices: list[int] = []
            for row_index in chunk_indices:
                if is_streaming_validation_index(
                    row_index,
                    validation_fraction=args.validation_fraction,
                ):
                    validation_count += 1
                else:
                    train_indices.append(row_index)
            if args.shuffle:
                rng.shuffle(train_indices)
        train_count += len(train_indices)
        batch_iter = iter(
            _index_batches(tuple(train_indices), batch_size=args.batch_size)
        )
        while True:
            try:
                with epoch_timings.time_phase("row_batching"):
                    row_indices = next(batch_iter)
            except StopIteration:
                break
            batch_count += 1
            with epoch_timings.time_phase("row_to_sample_batch"):
                sample_batch = flat_cache_batch(
                    cache=cache,
                    row_indices=row_indices,
                    requested_feature_names=args.feature_names,
                )
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
                    chunk_end,
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


def evaluate_flat_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: FlatTensorCache,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> RegressionEvaluationStats:
    """Compute streaming metrics for one split from cached flat tensors."""
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    target_count = 0
    row_limit = _cache_row_limit(cache=cache, max_rows=max_rows)
    model.eval()
    with torch.no_grad():
        for chunk_start in range(0, row_limit, row_chunk_size):
            with timings.time_phase("chunk_load"):
                chunk_end = min(chunk_start + row_chunk_size, row_limit)
                chunk_indices = tuple(range(chunk_start, chunk_end))
            with timings.time_phase("split_select"):
                selected_indices = tuple(
                    row_index
                    for row_index in chunk_indices
                    if streaming_row_is_in_split(
                        row_index,
                        validation_fraction=args.validation_fraction,
                        split=split,
                    )
                )
            batch_iter = iter(
                _index_batches(selected_indices, batch_size=args.batch_size)
            )
            while True:
                try:
                    with timings.time_phase("row_batching"):
                        row_indices = next(batch_iter)
                except StopIteration:
                    break
                with timings.time_phase("row_to_sample_batch"):
                    sample_batch = flat_cache_batch(
                        cache=cache,
                        row_indices=row_indices,
                        requested_feature_names=args.feature_names,
                    )
                sample_count += infer_batch_sample_count(sample_batch)
                batch_metrics = evaluate_regression_batch(
                    model=model,
                    batch=sample_batch,
                    device=device,
                    timings=timings,
                )
                squared_error_sum += batch_metrics.squared_error_sum
                absolute_error_sum += batch_metrics.absolute_error_sum
                target_count += batch_metrics.target_count
    elapsed_seconds = perf_counter() - evaluation_started_at
    if target_count == 0:
        return RegressionEvaluationStats(
            loss=0.0,
            mae=0.0,
            sample_count=0,
            target_count=0,
            phase_durations=timings.as_dict(),
            elapsed_seconds=elapsed_seconds,
        )
    return RegressionEvaluationStats(
        loss=squared_error_sum / target_count,
        mae=absolute_error_sum / target_count,
        sample_count=sample_count,
        target_count=target_count,
        phase_durations=timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def _index_batches(
    indices: tuple[int, ...],
    *,
    batch_size: int,
) -> Iterator[tuple[int, ...]]:
    """Yield fixed-size row-index batches."""
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if batch:
            yield batch


def _cache_row_limit(*, cache: FlatTensorCache, max_rows: int | None) -> int:
    """Return the effective cache row limit for one streaming pass."""
    if max_rows is None:
        return cache.manifest.row_count
    return min(cache.manifest.row_count, max_rows)


__all__ = [
    "evaluate_flat_cache_streaming_metrics",
    "train_flat_cache_streaming_epoch",
]
