"""Regression quality diagnostics for Morpion cached training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from chipiron.learning.supervised import (
    RegressionQualityStats,
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
    regression_quality_stats,
)

from .cached_index_schedule import index_batches

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class SplitQualityDiagnostics:
    """Regression diagnostics for one sampled split."""

    split: str
    sample_count: int
    stats: RegressionQualityStats


def cached_split_regression_quality_diagnostics(
    *,
    model: torch.nn.Module,
    batch_builder: Callable[[tuple[int, ...]], TensorSupervisedBatch],
    row_indices: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    max_rows: int,
    split: str,
) -> SplitQualityDiagnostics:
    """Return sampled prediction/target quality diagnostics for a cached split."""
    sampled_indices = _evenly_spaced_indices(row_indices, max_rows=max_rows)
    effective_batch_size = max(1, min(batch_size, 64))
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_indices in index_batches(
            sampled_indices,
            batch_size=effective_batch_size,
        ):
            sample_batch = batch_builder(batch_indices)
            device_batch = move_supervised_batch_to_device(sample_batch, device)
            batch_predictions = model(device_batch.get_input_layer())
            predictions.append(batch_predictions.detach().cpu().reshape(-1))
            targets.append(device_batch.get_target_value().detach().cpu().reshape(-1))
    if was_training:
        model.train()
    if not predictions:
        stats = regression_quality_stats(
            predictions=torch.empty((0,), dtype=torch.float32),
            targets=torch.empty((0,), dtype=torch.float32),
        )
    else:
        stats = regression_quality_stats(
            predictions=torch.cat(predictions),
            targets=torch.cat(targets),
        )
    return SplitQualityDiagnostics(
        split=split,
        sample_count=stats.count,
        stats=stats,
    )


def _evenly_spaced_indices(
    row_indices: tuple[int, ...],
    *,
    max_rows: int,
) -> tuple[int, ...]:
    """Return up to max_rows indices spaced across one ordered split."""
    if max_rows <= 0 or len(row_indices) <= max_rows:
        return row_indices
    return tuple(
        row_indices[index * len(row_indices) // max_rows] for index in range(max_rows)
    )


__all__ = [
    "SplitQualityDiagnostics",
    "cached_split_regression_quality_diagnostics",
]
