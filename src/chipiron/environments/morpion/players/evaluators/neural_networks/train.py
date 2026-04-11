"""Minimal supervised training helper for Morpion regressors."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
)

from .bundle import save_morpion_model_bundle
from .model import MorpionRegressor, MorpionRegressorArgs, build_morpion_regressor


@dataclass(frozen=True, slots=True)
class MorpionTrainingArgs:
    """Arguments for the first Morpion supervised-regression training helper."""

    dataset_file: str | os.PathLike[str]
    output_dir: str | os.PathLike[str]
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    hidden_dim: int | None = None


def train_morpion_regressor(
    args: MorpionTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float]]:
    """Train a small Morpion regressor on persisted supervised rows."""
    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(file_name=os.fspath(args.dataset_file))
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    model_args = MorpionRegressorArgs(
        model_kind=args.model_kind,
        input_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
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
        "hidden_dim": args.hidden_dim,
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


__all__ = [
    "MorpionTrainingArgs",
    "train_morpion_regressor",
]
