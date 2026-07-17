"""Streaming training loops backed by cached Morpion entity-token tensors."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import torch

from chipiron.learning.supervised import (
    RegressionEvaluationStats,
    TensorSupervisedBatch,
    evaluate_regression_batch,
    infer_batch_sample_count,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations

from .cached_index_schedule import (
    cached_index_schedule,
    index_batches,
    shuffled_epoch_train_indices,
    split_indices_for_streaming_policy,
)
from .entity_token_cache import MorpionEntityTokenCache, entity_token_cache_batch
from .relational_entity_token_cache import (
    MorpionRelationalEntityTokenCache,
    relational_entity_token_cache_batch,
)
from .streaming import StreamingEpochStats

if TYPE_CHECKING:
    from collections.abc import Callable

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .args import MorpionTrainingArgs

LOGGER = logging.getLogger(__name__)


def train_entity_token_cache_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    cache: MorpionEntityTokenCache,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached entity-token artifact."""
    return _train_packed_cache_streaming_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        args=args,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        cache_mode="entity_token_cache",
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        epoch_index=epoch_index,
        progress_callback=progress_callback,
        device=device,
    )


def train_relational_entity_token_cache_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    cache: MorpionRelationalEntityTokenCache,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached relational artifact."""
    return _train_packed_cache_streaming_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        args=args,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: relational_entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        cache_mode="relational_entity_token_cache",
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        epoch_index=epoch_index,
        progress_callback=progress_callback,
        device=device,
    )


def _train_packed_cache_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    row_count: int,
    batch_builder: Callable[[tuple[int, ...]], TensorSupervisedBatch],
    cache_mode: str,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> StreamingEpochStats:
    """Train one epoch using a packed cache batch builder."""
    epoch_started_at = perf_counter()
    epoch_timings = PhaseDurations()
    model.train()
    with epoch_timings.time_phase("chunk_load"):
        row_limit = _cache_row_limit(row_count=row_count, max_rows=max_rows)
    chunk_count = _pseudo_chunk_count(
        row_count=row_limit, row_chunk_size=row_chunk_size
    )
    batch_count = 0
    squared_error_sum = 0.0
    value_count = 0
    with epoch_timings.time_phase("split_select"):
        schedule = cached_index_schedule(
            row_count=row_limit,
            validation_fraction=args.validation_fraction,
        )
        epoch_train_indices = shuffled_epoch_train_indices(
            train_indices=schedule.train_indices,
            shuffle=args.shuffle,
            validation_seed=args.validation_seed,
            epoch_index=epoch_index,
        )
    LOGGER.info(
        "[train-schedule] mode=%s epoch=%s train_indices=%s "
        "validation_indices=%s shuffle=%s global_shuffle=%s split_policy=%s",
        cache_mode,
        epoch_index + 1,
        len(schedule.train_indices),
        len(schedule.validation_indices),
        str(args.shuffle).lower(),
        str(args.shuffle).lower(),
        schedule.split_policy,
    )
    processed_train_rows = 0
    reported_chunks = 0
    batch_iter = iter(index_batches(epoch_train_indices, batch_size=args.batch_size))
    while True:
        try:
            with epoch_timings.time_phase("row_batching"):
                row_indices = next(batch_iter)
        except StopIteration:
            break
        batch_count += 1
        with epoch_timings.time_phase("row_to_sample_batch"):
            sample_batch = batch_builder(row_indices)
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
        processed_train_rows += len(row_indices)
        if progress_callback is not None:
            with epoch_timings.time_phase("progress_callback"):
                reported_chunks = _report_cached_progress(
                    progress_callback=progress_callback,
                    reported_chunks=reported_chunks,
                    chunk_count=chunk_count,
                    processed_train_rows=processed_train_rows,
                    train_count=len(schedule.train_indices),
                    row_limit=row_limit,
                    epoch_number=epoch_index + 1,
                    num_epochs=args.num_epochs,
                )
    elapsed_seconds = perf_counter() - epoch_started_at
    epoch_timings.add_duration("epoch_total", elapsed_seconds)
    loss_value = 0.0 if value_count == 0 else squared_error_sum / value_count
    return StreamingEpochStats(
        chunk_count=chunk_count,
        train_count=len(schedule.train_indices),
        validation_count=len(schedule.validation_indices),
        batch_count=batch_count,
        loss=loss_value,
        phase_durations=epoch_timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_entity_token_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: MorpionEntityTokenCache,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> RegressionEvaluationStats:
    """Compute streaming metrics for one split from cached entity tokens."""
    return _evaluate_packed_cache_streaming_metrics(
        model=model,
        args=args,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        split=split,
        device=device,
    )


def evaluate_relational_entity_token_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: MorpionRelationalEntityTokenCache,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> RegressionEvaluationStats:
    """Compute streaming metrics from cached relational entity tokens."""
    return _evaluate_packed_cache_streaming_metrics(
        model=model,
        args=args,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: relational_entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        split=split,
        device=device,
    )


def _evaluate_packed_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    row_count: int,
    batch_builder: Callable[[tuple[int, ...]], TensorSupervisedBatch],
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> RegressionEvaluationStats:
    """Compute streaming metrics using a packed cache batch builder."""
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    target_count = 0
    row_limit = _cache_row_limit(row_count=row_count, max_rows=max_rows)
    model.eval()
    with torch.no_grad():
        with timings.time_phase("chunk_load"):
            selected_row_count = row_limit
        with timings.time_phase("split_select"):
            selected_indices = split_indices_for_streaming_policy(
                row_count=selected_row_count,
                validation_fraction=args.validation_fraction,
                split=split,
            )
        batch_iter = iter(index_batches(selected_indices, batch_size=args.batch_size))
        while True:
            try:
                with timings.time_phase("row_batching"):
                    row_indices = next(batch_iter)
            except StopIteration:
                break
            with timings.time_phase("row_to_sample_batch"):
                sample_batch = batch_builder(row_indices)
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


def _cache_row_limit(*, row_count: int, max_rows: int | None) -> int:
    """Return the effective cache row limit for one streaming pass."""
    if max_rows is None:
        return row_count
    return min(row_count, max_rows)


def _pseudo_chunk_count(*, row_count: int, row_chunk_size: int) -> int:
    """Return a cache-compatible pseudo-chunk count for progress and logs."""
    if row_count <= 0:
        return 0
    return math.ceil(row_count / row_chunk_size)


def _report_cached_progress(
    *,
    progress_callback: Callable[[int, int, int, int], None],
    reported_chunks: int,
    chunk_count: int,
    processed_train_rows: int,
    train_count: int,
    row_limit: int,
    epoch_number: int,
    num_epochs: int,
) -> int:
    """Report approximate chunk progress for globally shuffled cache training."""
    if chunk_count == 0:
        return reported_chunks
    while reported_chunks < chunk_count:
        next_chunk = reported_chunks + 1
        threshold = math.ceil(train_count * next_chunk / chunk_count)
        if processed_train_rows < threshold:
            break
        progress_callback(
            next_chunk,
            min(next_chunk * row_limit // chunk_count, row_limit),
            epoch_number,
            num_epochs,
        )
        reported_chunks = next_chunk
    return reported_chunks


__all__ = [
    "evaluate_entity_token_cache_streaming_metrics",
    "evaluate_relational_entity_token_cache_streaming_metrics",
    "train_entity_token_cache_streaming_epoch",
    "train_relational_entity_token_cache_streaming_epoch",
]
