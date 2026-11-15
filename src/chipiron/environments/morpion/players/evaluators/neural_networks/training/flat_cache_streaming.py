"""Streaming training loops backed by cached Morpion flat tensors."""

import logging
from typing import TYPE_CHECKING, Literal

from chipiron.learning.supervised.cached import (
    evaluate_cached_metrics,
    train_cached_epoch,
)

from .flat_tensor_cache import FlatTensorCache, flat_cache_batch

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )
    from chipiron.learning.supervised import (
        RegressionEvaluationStats,
    )

    from .args import MorpionTrainingArgs
    from .streaming import StreamingEpochStats

LOGGER = logging.getLogger(__name__)


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
    synchronize_cuda: bool = True,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached Morpion flat tensor artifact."""
    return train_cached_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        shuffle=args.shuffle,
        validation_seed=args.validation_seed,
        num_epochs=args.num_epochs,
        logger=LOGGER,
        explicit_split_policy="canonical_state_sha256_v1",
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: flat_cache_batch(
            cache=cache,
            row_indices=row_indices,
            requested_feature_names=args.feature_names,
        ),
        cache_mode="flat_cache",
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        epoch_index=epoch_index,
        progress_callback=progress_callback,
        device=device,
        synchronize_cuda=synchronize_cuda,
    )


def evaluate_flat_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: FlatTensorCache,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
    synchronize_cuda: bool = True,
) -> RegressionEvaluationStats:
    """Compute streaming metrics for one split from cached flat tensors."""
    return evaluate_cached_metrics(
        model=model,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: flat_cache_batch(
            cache=cache,
            row_indices=row_indices,
            requested_feature_names=args.feature_names,
        ),
        max_rows=max_rows,
        split=split,
        device=device,
        synchronize_cuda=synchronize_cuda,
    )


__all__ = [
    "evaluate_flat_cache_streaming_metrics",
    "train_flat_cache_streaming_epoch",
]
