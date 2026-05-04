"""Minimal supervised training helper for Morpion regressors."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)

from .bundle import save_morpion_model_bundle
from .model import MorpionRegressor, MorpionRegressorArgs, build_morpion_regressor


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
    validation_fraction: float = 0.2
    validation_seed: int = 0

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if (
            not math.isfinite(self.validation_fraction)
            or self.validation_fraction < 0.0
            or self.validation_fraction >= 1.0
        ):
            raise ValueError("validation_fraction must be in [0.0, 1.0).")
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


def train_morpion_regressor(
    args: MorpionTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor on persisted supervised rows."""
    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(
            file_name=os.fspath(args.dataset_file),
            feature_subset_name=args.feature_subset_name,
            feature_names=args.feature_names,
        )
    )
    train_dataset, validation_dataset = _split_train_validation_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )

    resolved_hidden_sizes = _resolve_hidden_sizes(args)
    model_args = MorpionRegressorArgs(
        model_kind=args.model_kind,
        feature_subset_name=args.feature_subset_name,
        feature_names=args.feature_names,
        hidden_sizes=resolved_hidden_sizes,
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
    )
    validation_loss: float | None
    validation_mae: float | None
    if len(validation_dataset) > 0:
        validation_loss, validation_mae = _evaluate_regression_metrics(
            model,
            validation_dataset,
            batch_size=args.batch_size,
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


def _split_train_validation_dataset(
    dataset: MorpionSupervisedDataset,
    *,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[Subset[MorpionSupervisedDataset], Subset[MorpionSupervisedDataset]]:
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


def _evaluate_regression_metrics(
    model: MorpionRegressor,
    dataset: Dataset[Any],
    *,
    batch_size: int,
) -> tuple[float, float]:
    """Compute full-dataset mean MSE and MAE for one regression split."""
    if len(dataset) == 0:
        return 0.0, 0.0

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    "MorpionTrainingArgs",
    "train_morpion_regressor",
]
