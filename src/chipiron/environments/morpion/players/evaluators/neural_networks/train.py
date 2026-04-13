"""Minimal supervised training helper for Morpion regressors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

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

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
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
) -> tuple[MorpionRegressor, dict[str, float]]:
    """Train a Morpion regressor on persisted supervised rows."""
    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(
            file_name=os.fspath(args.dataset_file),
            feature_subset_name=args.feature_subset_name,
            feature_names=args.feature_names,
        )
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

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

    final_loss = 0.0
    model.train()
    for _epoch in range(args.num_epochs):
        for sample_batch in data_loader:
            optimizer.zero_grad()
            predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    metrics: dict[str, float] = {
        "final_loss": final_loss,
        "num_samples": float(len(dataset)),
        "num_epochs": float(args.num_epochs),
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
