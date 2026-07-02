"""Morpion regression evaluation passes."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.data import DataLoader, Subset

from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    MorpionGraphSupervisedDataset,
    MorpionGraphSupervisedSample,
    MorpionSupervisedDataset,
    MorpionSupervisedSample,
)
from chipiron.learning.supervised import (
    RegressionEvaluationStats,
    evaluate_regression_batch,
    infer_batch_sample_count,
)
from chipiron.learning.timing import PhaseDurations

from .row_batches import indexed_row_chunks, row_batches, rows_to_sample_batch
from .streaming import streaming_row_is_in_split

if TYPE_CHECKING:
    from collections.abc import Callable

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .args import MorpionTrainingArgs

type MorpionRegressionDataset = (
    MorpionSupervisedDataset
    | MorpionGraphSupervisedDataset
    | Subset[MorpionSupervisedSample]
    | Subset[MorpionGraphSupervisedSample]
)


def evaluate_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> RegressionEvaluationStats:
    """Compute streaming mean MSE and MAE for one regression split."""
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    target_count = 0
    model.eval()
    with torch.no_grad():
        row_chunk_iter = iter(
            indexed_row_chunks(
                args.dataset_file,
                chunk_size=row_chunk_size,
                max_rows=max_rows,
            )
        )
        while True:
            try:
                with timings.time_phase("chunk_load"):
                    row_start_index, rows = next(row_chunk_iter)
            except StopIteration:
                break
            with timings.time_phase("split_select"):
                selected_rows = [
                    row
                    for offset, row in enumerate(rows)
                    if streaming_row_is_in_split(
                        row_start_index + offset,
                        validation_fraction=args.validation_fraction,
                        split=split,
                    )
                ]
            row_batch_iter = iter(
                row_batches(selected_rows, batch_size=args.batch_size)
            )
            while True:
                try:
                    with timings.time_phase("row_batching"):
                        row_batch = next(row_batch_iter)
                except StopIteration:
                    break
                with timings.time_phase("row_to_sample_batch"):
                    sample_batch = rows_to_sample_batch(row_batch, args=args)
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


def evaluate_regression_metrics(
    model: MorpionRegressor,
    dataset: MorpionRegressionDataset,
    *,
    batch_size: int,
    device: torch.device,
    collate_fn: Callable[[Any], Any] | None = None,
) -> RegressionEvaluationStats:
    """Compute full-dataset mean MSE and MAE for one regression split."""
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    if len(dataset) == 0:
        return RegressionEvaluationStats(
            loss=0.0,
            mae=0.0,
            sample_count=0,
            target_count=0,
            phase_durations={},
            elapsed_seconds=0.0,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    target_count = 0
    model.eval()
    with torch.no_grad():
        for sample_batch in data_loader:
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
