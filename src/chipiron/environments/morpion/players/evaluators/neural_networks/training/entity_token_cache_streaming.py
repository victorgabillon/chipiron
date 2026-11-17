"""Streaming training loops backed by cached Morpion entity-token tensors."""

import logging
from typing import TYPE_CHECKING, Literal

from chipiron.learning.supervised.cached import (
    evaluate_cached_metrics,
    train_cached_epoch,
)

from .entity_token_cache import MorpionEntityTokenCache, entity_token_cache_batch
from .relational_entity_token_cache import (
    MorpionRelationalEntityTokenCache,
    relational_entity_token_cache_batch,
)

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
    synchronize_cuda: bool = True,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached entity-token artifact."""
    return train_cached_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        shuffle=args.shuffle,
        validation_seed=args.validation_seed,
        num_epochs=args.num_epochs,
        logger=LOGGER,
        explicit_split_policy="canonical_state_sha256_v1",
        cache_mode="entity_token_cache",
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        epoch_index=epoch_index,
        progress_callback=progress_callback,
        device=device,
        synchronize_cuda=synchronize_cuda,
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
    synchronize_cuda: bool = True,
) -> StreamingEpochStats:
    """Train one streaming epoch over a cached relational artifact."""
    return train_cached_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: relational_entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        shuffle=args.shuffle,
        validation_seed=args.validation_seed,
        num_epochs=args.num_epochs,
        logger=LOGGER,
        explicit_split_policy="canonical_state_sha256_v1",
        cache_mode="relational_entity_token_cache",
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        epoch_index=epoch_index,
        progress_callback=progress_callback,
        device=device,
        synchronize_cuda=synchronize_cuda,
    )


def evaluate_entity_token_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: MorpionEntityTokenCache,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
    synchronize_cuda: bool = True,
) -> RegressionEvaluationStats:
    """Compute streaming metrics for one split from cached entity tokens."""
    return evaluate_cached_metrics(
        model=model,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        max_rows=max_rows,
        split=split,
        device=device,
        synchronize_cuda=synchronize_cuda,
    )


def evaluate_relational_entity_token_cache_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    cache: MorpionRelationalEntityTokenCache,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
    synchronize_cuda: bool = True,
) -> RegressionEvaluationStats:
    """Compute streaming metrics from cached relational entity tokens."""
    return evaluate_cached_metrics(
        model=model,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        validation_indices=args.validation_indices,
        row_count=cache.manifest.row_count,
        batch_builder=lambda row_indices: relational_entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        ),
        max_rows=max_rows,
        split=split,
        device=device,
        synchronize_cuda=synchronize_cuda,
    )


__all__ = [
    "evaluate_entity_token_cache_streaming_metrics",
    "evaluate_relational_entity_token_cache_streaming_metrics",
    "train_entity_token_cache_streaming_epoch",
    "train_relational_entity_token_cache_streaming_epoch",
]
