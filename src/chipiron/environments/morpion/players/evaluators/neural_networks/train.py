"""Minimal supervised training helper for Morpion regressors."""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.data import DataLoader, Subset

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    iter_morpion_supervised_row_chunks_from_path,
)
from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    MorpionGraphSupervisedDataset,
    MorpionGraphSupervisedDatasetArgs,
    MorpionGraphSupervisedSample,
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
    MorpionSupervisedSample,
    collate_morpion_graph_supervised_samples,
    process_morpion_supervised_row_to_graph_tensors,
    process_morpion_supervised_row_to_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_MODEL_KIND,
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MorpionGraphTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

from .bundle import save_morpion_model_bundle
from .model import MorpionRegressor, MorpionRegressorArgs, build_morpion_regressor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

LOGGER = logging.getLogger(__name__)

type MorpionRegressionDataset = (
    MorpionSupervisedDataset
    | MorpionGraphSupervisedDataset
    | Subset[MorpionSupervisedSample]
    | Subset[MorpionGraphSupervisedSample]
)


class InvalidValidationFractionError(ValueError):
    """Raised when supervised-training validation splitting is outside bounds."""

    def __init__(self) -> None:
        """Initialize the invalid-validation-fraction error."""
        super().__init__("validation_fraction must be in [0.0, 1.0).")


@dataclass(frozen=True, slots=True)
class MorpionTrainingArgs:
    """Arguments for the Morpion supervised-regression training helper."""

    dataset_file: str | os.PathLike[str]
    output_dir: str | os.PathLike[str]
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_sizes: tuple[int, ...] | None = None
    hidden_dim: int | None = None
    graph_max_tokens: int = 1536
    graph_input_feature_dim: int = MORPION_GRAPH_TOKEN_FEATURE_DIM
    graph_d_model: int = 64
    graph_n_head: int = 4
    graph_n_layer: int = 2
    graph_dim_feedforward: int = 256
    graph_dropout_ratio: float = 0.0
    graph_pooling: str = "value_token"
    graph_output_tanh: bool = True
    validation_fraction: float = 0.2
    validation_seed: int = 0

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if (
            not math.isfinite(self.validation_fraction)
            or self.validation_fraction < 0.0
            or self.validation_fraction >= 1.0
        ):
            raise InvalidValidationFractionError
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this training job."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionStreamingTrainingArgs:
    """Arguments for streaming Morpion supervised-regression training."""

    training_args: MorpionTrainingArgs
    row_chunk_size: int = 8192
    max_rows: int | None = None

    def __post_init__(self) -> None:
        """Validate streaming controls."""
        if isinstance(self.row_chunk_size, bool) or self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be a positive integer.")  # noqa: TRY003
        if self.max_rows is not None and (
            isinstance(self.max_rows, bool) or self.max_rows < 0
        ):
            raise ValueError("max_rows must be a non-negative integer or None.")  # noqa: TRY003


def train_morpion_regressor(
    args: MorpionTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor on persisted supervised rows."""
    collate_fn: Callable[[Any], Any] | None
    if args.model_kind == MORPION_GRAPH_MODEL_KIND:
        dataset = MorpionGraphSupervisedDataset(
            MorpionGraphSupervisedDatasetArgs(
                file_name=os.fspath(args.dataset_file),
                max_tokens=args.graph_max_tokens,
            )
        )
        collate_fn = collate_morpion_graph_supervised_samples
    else:
        dataset = MorpionSupervisedDataset(
            MorpionSupervisedDatasetArgs(
                file_name=os.fspath(args.dataset_file),
                feature_subset_name=args.feature_subset_name,
                feature_names=args.feature_names,
            )
        )
        collate_fn = None
    train_dataset, validation_dataset = _split_train_validation_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_fn,
    )

    resolved_hidden_sizes = _resolve_hidden_sizes(args)
    model_args = MorpionRegressorArgs(
        model_kind=args.model_kind,
        feature_subset_name=args.feature_subset_name,
        feature_names=args.feature_names,
        hidden_sizes=resolved_hidden_sizes,
        graph_max_tokens=args.graph_max_tokens,
        graph_input_feature_dim=args.graph_input_feature_dim,
        graph_d_model=args.graph_d_model,
        graph_n_head=args.graph_n_head,
        graph_n_layer=args.graph_n_layer,
        graph_dim_feedforward=args.graph_dim_feedforward,
        graph_dropout_ratio=args.graph_dropout_ratio,
        graph_pooling=args.graph_pooling,
        graph_output_tanh=args.graph_output_tanh,
    )
    model = build_morpion_regressor(model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    model.train()
    for _epoch in range(args.num_epochs):
        for sample_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

    train_loss, train_mae = _evaluate_regression_metrics(
        model,
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    validation_loss: float | None
    validation_mae: float | None
    if len(validation_dataset) > 0:
        validation_loss, validation_mae = _evaluate_regression_metrics(
            model,
            validation_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
    else:
        validation_loss = None
        validation_mae = None
    final_loss = validation_loss if validation_loss is not None else train_loss

    metrics: dict[str, float | str | None] = {
        "final_loss": final_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_mae": train_mae,
        "validation_mae": validation_mae,
        "num_samples": float(len(dataset)),
        "num_train_samples": float(len(train_dataset)),
        "num_validation_samples": float(len(validation_dataset)),
        "num_epochs": float(args.num_epochs),
        "batch_size": float(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "loss_name": "mse",
    }
    training_metadata = {
        "dataset_file": os.fspath(args.dataset_file),
        "output_dir": os.fspath(args.output_dir),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "shuffle": args.shuffle,
        "model_kind": args.model_kind,
        "feature_subset_name": args.feature_subset_name,
        "feature_names": args.feature_names,
        "input_dim": model_args.input_dim,
        "hidden_sizes": resolved_hidden_sizes,
        "graph_max_tokens": args.graph_max_tokens,
        "graph_input_feature_dim": args.graph_input_feature_dim,
        "graph_d_model": args.graph_d_model,
        "graph_n_head": args.graph_n_head,
        "graph_n_layer": args.graph_n_layer,
        "graph_dim_feedforward": args.graph_dim_feedforward,
        "graph_dropout_ratio": args.graph_dropout_ratio,
        "graph_pooling": args.graph_pooling,
        "graph_output_tanh": args.graph_output_tanh,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
    }
    save_morpion_model_bundle(
        model,
        os.fspath(args.output_dir),
        model_args=model_args,
        metadata={
            **training_metadata,
            **metrics,
        },
    )
    return model, metrics


def train_morpion_regressor_streaming(
    args: MorpionStreamingTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor from row chunks without materializing all rows."""
    training_args = args.training_args
    resolved_hidden_sizes = _resolve_hidden_sizes(training_args)
    model_args = MorpionRegressorArgs(
        model_kind=training_args.model_kind,
        feature_subset_name=training_args.feature_subset_name,
        feature_names=training_args.feature_names,
        hidden_sizes=resolved_hidden_sizes,
        graph_max_tokens=training_args.graph_max_tokens,
        graph_input_feature_dim=training_args.graph_input_feature_dim,
        graph_d_model=training_args.graph_d_model,
        graph_n_head=training_args.graph_n_head,
        graph_n_layer=training_args.graph_n_layer,
        graph_dim_feedforward=training_args.graph_dim_feedforward,
        graph_dropout_ratio=training_args.graph_dropout_ratio,
        graph_pooling=training_args.graph_pooling,
        graph_output_tanh=training_args.graph_output_tanh,
    )
    model = build_morpion_regressor(model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    criterion = torch.nn.MSELoss()
    split_policy = _streaming_split_policy(training_args.validation_fraction)

    for epoch_index in range(training_args.num_epochs):
        epoch_stats = _train_streaming_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            args=training_args,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            epoch_index=epoch_index,
        )
        LOGGER.info(
            "[train-stream] "
            f"epoch={epoch_index + 1} chunks={epoch_stats.chunk_count} "
            f"train_samples={epoch_stats.train_count} "
            f"validation_samples={epoch_stats.validation_count} "
            f"train_loss={epoch_stats.loss} split_policy={split_policy}",
        )

    train_loss, train_mae, train_count = _evaluate_streaming_metrics(
        model=model,
        args=training_args,
        row_chunk_size=args.row_chunk_size,
        max_rows=args.max_rows,
        split="train",
    )
    validation_loss: float | None
    validation_mae: float | None
    validation_loss_value, validation_mae_value, validation_count = (
        _evaluate_streaming_metrics(
            model=model,
            args=training_args,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            split="validation",
        )
    )
    if validation_count > 0:
        validation_loss = validation_loss_value
        validation_mae = validation_mae_value
    else:
        validation_loss = None
        validation_mae = None
    num_samples = train_count + validation_count
    final_loss = validation_loss if validation_loss is not None else train_loss
    metrics: dict[str, float | str | None] = {
        "final_loss": final_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_mae": train_mae,
        "validation_mae": validation_mae,
        "num_samples": float(num_samples),
        "num_train_samples": float(train_count),
        "num_validation_samples": float(validation_count),
        "num_epochs": float(training_args.num_epochs),
        "batch_size": float(training_args.batch_size),
        "learning_rate": float(training_args.learning_rate),
        "loss_name": "mse",
        "split_policy": split_policy,
    }
    training_metadata = {
        "dataset_file": os.fspath(training_args.dataset_file),
        "output_dir": os.fspath(training_args.output_dir),
        "batch_size": training_args.batch_size,
        "num_epochs": training_args.num_epochs,
        "learning_rate": training_args.learning_rate,
        "shuffle": training_args.shuffle,
        "model_kind": training_args.model_kind,
        "feature_subset_name": training_args.feature_subset_name,
        "feature_names": training_args.feature_names,
        "input_dim": model_args.input_dim,
        "hidden_sizes": resolved_hidden_sizes,
        "graph_max_tokens": training_args.graph_max_tokens,
        "graph_input_feature_dim": training_args.graph_input_feature_dim,
        "graph_d_model": training_args.graph_d_model,
        "graph_n_head": training_args.graph_n_head,
        "graph_n_layer": training_args.graph_n_layer,
        "graph_dim_feedforward": training_args.graph_dim_feedforward,
        "graph_dropout_ratio": training_args.graph_dropout_ratio,
        "graph_pooling": training_args.graph_pooling,
        "graph_output_tanh": training_args.graph_output_tanh,
        "validation_fraction": training_args.validation_fraction,
        "validation_seed": training_args.validation_seed,
        "streaming": True,
        "row_chunk_size": args.row_chunk_size,
        "max_rows": args.max_rows,
        "split_policy": split_policy,
    }
    save_morpion_model_bundle(
        model,
        os.fspath(training_args.output_dir),
        model_args=model_args,
        metadata={
            **training_metadata,
            **metrics,
        },
    )
    return model, metrics


def _split_train_validation_dataset(
    dataset: MorpionSupervisedDataset | MorpionGraphSupervisedDataset,
    *,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[
    Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample],
    Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample],
]:
    """Return deterministic train/validation subsets for one supervised dataset."""
    sample_count = len(dataset)
    indices = list(range(sample_count))
    if sample_count < 2 or validation_fraction <= 0.0:
        return Subset(dataset, indices), Subset(dataset, [])

    rng = random.Random(validation_seed)
    rng.shuffle(indices)
    validation_count = max(1, round(sample_count * validation_fraction))
    validation_count = min(sample_count - 1, validation_count)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


@dataclass(frozen=True, slots=True)
class _StreamingEpochStats:
    chunk_count: int
    train_count: int
    validation_count: int
    loss: float


def _train_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
) -> _StreamingEpochStats:
    model.train()
    chunk_count = 0
    train_count = 0
    validation_count = 0
    squared_error_sum = 0.0
    value_count = 0
    rng = random.Random(args.validation_seed + epoch_index)
    for row_start_index, rows in _indexed_row_chunks(
        args.dataset_file,
        chunk_size=row_chunk_size,
        max_rows=max_rows,
    ):
        chunk_count += 1
        train_rows: list[MorpionSupervisedRow] = []
        for offset, row in enumerate(rows):
            row_index = row_start_index + offset
            if _is_streaming_validation_index(
                row_index,
                validation_fraction=args.validation_fraction,
            ):
                validation_count += 1
            else:
                train_rows.append(row)
        if args.shuffle:
            rng.shuffle(train_rows)
        train_count += len(train_rows)
        for row_batch in _row_batches(train_rows, batch_size=args.batch_size):
            sample_batch = _rows_to_sample_batch(row_batch, args=args)
            optimizer.zero_grad()
            predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            errors = predictions.detach() - targets
            squared_error_sum += float(torch.sum(errors * errors).item())
            value_count += int(targets.numel())
    loss_value = 0.0 if value_count == 0 else squared_error_sum / value_count
    return _StreamingEpochStats(
        chunk_count=chunk_count,
        train_count=train_count,
        validation_count=validation_count,
        loss=loss_value,
    )


def _evaluate_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
) -> tuple[float, float, int]:
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    model.eval()
    with torch.no_grad():
        for row_start_index, rows in _indexed_row_chunks(
            args.dataset_file,
            chunk_size=row_chunk_size,
            max_rows=max_rows,
        ):
            selected_rows = [
                row
                for offset, row in enumerate(rows)
                if _streaming_row_is_in_split(
                    row_start_index + offset,
                    validation_fraction=args.validation_fraction,
                    split=split,
                )
            ]
            for row_batch in _row_batches(selected_rows, batch_size=args.batch_size):
                sample_batch = _rows_to_sample_batch(row_batch, args=args)
                predictions = model(sample_batch.get_input_layer())
                targets = sample_batch.get_target_value()
                errors = predictions - targets
                squared_error_sum += float(torch.sum(errors * errors).item())
                absolute_error_sum += float(torch.sum(torch.abs(errors)).item())
                value_count += int(targets.numel())
    if value_count == 0:
        return 0.0, 0.0, 0
    return squared_error_sum / value_count, absolute_error_sum / value_count, value_count


def _indexed_row_chunks(
    path: str | os.PathLike[str],
    *,
    chunk_size: int,
    max_rows: int | None,
) -> Iterable[tuple[int, tuple[MorpionSupervisedRow, ...]]]:
    row_start_index = 0
    for rows in iter_morpion_supervised_row_chunks_from_path(
        path,
        chunk_size=chunk_size,
        max_rows=max_rows,
    ):
        yield row_start_index, rows
        row_start_index += len(rows)


def _streaming_row_is_in_split(
    row_index: int,
    *,
    validation_fraction: float,
    split: Literal["train", "validation"],
) -> bool:
    is_validation = _is_streaming_validation_index(
        row_index,
        validation_fraction=validation_fraction,
    )
    return is_validation if split == "validation" else not is_validation


def _is_streaming_validation_index(
    row_index: int,
    *,
    validation_fraction: float,
) -> bool:
    if validation_fraction <= 0.0:
        return False
    period = max(2, round(1.0 / validation_fraction))
    return (row_index + 1) % period == 0


def _streaming_split_policy(validation_fraction: float) -> str:
    if validation_fraction <= 0.0:
        return "none"
    return f"index_modulo_{max(2, round(1.0 / validation_fraction))}"


def _row_batches(
    rows: list[MorpionSupervisedRow],
    *,
    batch_size: int,
) -> Iterable[tuple[MorpionSupervisedRow, ...]]:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if batch:
            yield tuple(batch)


def _rows_to_sample_batch(
    rows: tuple[MorpionSupervisedRow, ...],
    *,
    args: MorpionTrainingArgs,
) -> MorpionSupervisedSample | MorpionGraphSupervisedSample:
    dynamics = MorpionDynamics()
    if args.model_kind == MORPION_GRAPH_MODEL_KIND:
        graph_converter = MorpionGraphTokenConverter(
            dynamics=dynamics,
            max_tokens=args.graph_max_tokens,
        )
        return collate_morpion_graph_supervised_samples(
            [
                process_morpion_supervised_row_to_graph_tensors(
                    row,
                    dynamics=dynamics,
                    converter=graph_converter,
                )
                for row in rows
            ]
        )
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
    return MorpionSupervisedSample(
        input_tensor=torch.stack([sample.input_tensor for sample in samples]),
        target_tensor=torch.stack([sample.target_tensor for sample in samples]),
    )


def _evaluate_regression_metrics(
    model: MorpionRegressor,
    dataset: MorpionRegressionDataset,
    *,
    batch_size: int,
    collate_fn: Callable[[Any], Any] | None = None,
) -> tuple[float, float]:
    """Compute full-dataset mean MSE and MAE for one regression split."""
    if len(dataset) == 0:
        return 0.0, 0.0

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    model.eval()
    with torch.no_grad():
        for sample_batch in data_loader:
            predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            errors = predictions - targets
            squared_error_sum += float(torch.sum(errors * errors).item())
            absolute_error_sum += float(torch.sum(torch.abs(errors)).item())
            value_count += int(targets.numel())
    if value_count == 0:
        return 0.0, 0.0
    return squared_error_sum / value_count, absolute_error_sum / value_count


def _resolve_hidden_sizes(args: MorpionTrainingArgs) -> tuple[int, ...] | None:
    """Resolve legacy and current hidden-layer arguments into one tuple."""
    if args.hidden_sizes is not None:
        return args.hidden_sizes
    if args.hidden_dim is not None:
        return (args.hidden_dim,)
    return None


__all__ = [
    "MorpionStreamingTrainingArgs",
    "MorpionTrainingArgs",
    "train_morpion_regressor",
    "train_morpion_regressor_streaming",
]
