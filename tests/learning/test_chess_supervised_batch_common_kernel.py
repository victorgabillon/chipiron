"""Tests for chess supervised batches with common learning kernels."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

from chipiron.learning.supervised import (
    evaluate_regression_batch,
    train_regression_batch,
)


def test_common_train_regression_batch_accepts_fen_and_value_data() -> None:
    """The common train kernel should work with the chess batch object."""
    _require_chess_learning_dependencies()
    datasets = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "datasets.datasets"
    )
    sample_1 = datasets.FenAndValueData(
        fen_tensor=torch.ones(5),
        value_tensor=torch.tensor([0.5], dtype=torch.float32),
    )
    sample_2 = datasets.FenAndValueData(
        fen_tensor=-torch.ones(5),
        value_tensor=torch.tensor([-0.5], dtype=torch.float32),
    )
    batch = datasets.custom_collate_fn_fen_and_value([sample_1, sample_2])
    model_inputs = batch.get_model_input_tensors()

    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    stats = train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch=batch,
        device=torch.device("cpu"),
        timings=None,
    )
    after = list(model.parameters())

    assert len(model_inputs) == 1
    assert model_inputs[0] is batch.get_input_layer()
    assert stats.target_count == 2
    assert stats.squared_error_sum >= 0.0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, after, strict=True)
    )


def test_common_evaluate_regression_batch_accepts_fen_and_value_data() -> None:
    """The common eval kernel should work with the chess batch object."""
    _require_chess_learning_dependencies()
    datasets = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "datasets.datasets"
    )
    sample_1 = datasets.FenAndValueData(
        fen_tensor=torch.ones(5),
        value_tensor=torch.tensor([0.5], dtype=torch.float32),
    )
    sample_2 = datasets.FenAndValueData(
        fen_tensor=-torch.ones(5),
        value_tensor=torch.tensor([-0.5], dtype=torch.float32),
    )
    batch = datasets.custom_collate_fn_fen_and_value([sample_1, sample_2])

    metrics = evaluate_regression_batch(
        model=nn.Linear(5, 1),
        batch=batch,
        device=torch.device("cpu"),
        timings=None,
    )

    assert metrics.target_count == 2
    assert metrics.squared_error_sum >= 0.0


def _require_chess_learning_dependencies() -> None:
    """Skip chess batch tests when optional runtime dependencies are absent."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")
