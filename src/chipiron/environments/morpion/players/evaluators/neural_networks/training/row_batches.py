"""Morpion supervised-row batching and tensor conversion helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    iter_morpion_supervised_row_chunks_from_path,
)
from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    collate_morpion_entity_token_supervised_samples,
    process_morpion_supervised_row_to_entity_token_tensors,
    process_morpion_supervised_row_to_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
    is_morpion_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .args import MorpionTrainingArgs


def indexed_row_chunks(
    path: str | os.PathLike[str],
    *,
    chunk_size: int,
    max_rows: int | None,
) -> Iterable[tuple[int, tuple[MorpionSupervisedRow, ...]]]:
    """Yield persisted row chunks with their absolute starting index."""
    row_start_index = 0
    row_path = os.fspath(path)
    for rows in iter_morpion_supervised_row_chunks_from_path(
        row_path,
        chunk_size=chunk_size,
        max_rows=max_rows,
    ):
        yield row_start_index, rows
        row_start_index += len(rows)


def row_batches(
    rows: list[MorpionSupervisedRow],
    *,
    batch_size: int,
) -> Iterable[tuple[MorpionSupervisedRow, ...]]:
    """Yield fixed-size supervised-row batches."""
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if batch:
            yield tuple(batch)


def rows_to_sample_batch(
    rows: tuple[MorpionSupervisedRow, ...],
    *,
    args: MorpionTrainingArgs,
) -> TensorSupervisedBatch:
    """Convert Morpion supervised rows into one tensor batch."""
    dynamics = MorpionDynamics()
    if is_morpion_entity_token_model_kind(args.model_kind):
        entity_converter = MorpionEntityTokenConverter(
            dynamics=dynamics,
            max_tokens=args.entity_max_tokens,
        )
        return collate_morpion_entity_token_supervised_samples([
            process_morpion_supervised_row_to_entity_token_tensors(
                row,
                dynamics=dynamics,
                converter=entity_converter,
            )
            for row in rows
        ])
    feature_converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=args.feature_subset,
    )
    samples = [
        process_morpion_supervised_row_to_tensors(
            row,
            dynamics=dynamics,
            converter=feature_converter,
        )
        for row in rows
    ]
    return TensorSupervisedBatch(
        input_tensor=torch.stack([sample.input_tensor for sample in samples]),
        target_tensor=torch.stack([sample.target_tensor for sample in samples]),
        is_batch=True,
    )
