"""Tests for common supervised regression batch helpers."""

from __future__ import annotations

import torch
from torch import nn

from chipiron.learning.supervised import TensorSupervisedBatch
from chipiron.learning.supervised.regression import (
    evaluate_regression_batch,
    infer_batch_sample_count,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations


class ConstantZeroPrediction(nn.Module):
    """Tiny model returning zero predictions for deterministic metrics."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return one zero scalar prediction per input row."""
        return torch.zeros((input_tensor.shape[0], 1), dtype=input_tensor.dtype)


def test_train_regression_batch_updates_model_parameters() -> None:
    """One regression train batch should update model parameters."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0]]),
        target_tensor=torch.tensor([[1.0]]),
        is_batch=True,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    stats = train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch=batch,
        device=torch.device("cpu"),
    )
    after = dict(model.named_parameters())

    assert stats.loss >= 0.0
    assert stats.sample_count == 1
    assert stats.target_count == 1
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in after.items()
    )


def test_train_regression_batch_records_timings() -> None:
    """One regression train batch should record stable timing phase names."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    timings = PhaseDurations()
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0]]),
        target_tensor=torch.tensor([[1.0]]),
        is_batch=True,
    )

    train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch=batch,
        device=torch.device("cpu"),
        timings=timings,
    )

    for phase in (
        "batch_transfer",
        "zero_grad",
        "forward",
        "loss",
        "backward",
        "optimizer_step",
        "metric_accumulation",
    ):
        assert timings.get(phase) >= 0.0


def test_evaluate_regression_batch_does_not_update_parameters() -> None:
    """One regression evaluation batch should not update model parameters."""
    model = nn.Linear(2, 1)
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0]]),
        target_tensor=torch.tensor([[1.0]]),
        is_batch=True,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    model.eval()
    with torch.no_grad():
        evaluate_regression_batch(
            model=model,
            batch=batch,
            device=torch.device("cpu"),
        )
    after = dict(model.named_parameters())

    assert all(
        torch.equal(before[name], parameter) for name, parameter in after.items()
    )


def test_evaluate_regression_batch_returns_exact_metric_sums() -> None:
    """Regression batch evaluation should return exact MSE and MAE sums."""
    model = ConstantZeroPrediction()
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        target_tensor=torch.tensor([[1.0], [-2.0]]),
        is_batch=True,
    )

    model.eval()
    with torch.no_grad():
        metrics = evaluate_regression_batch(
            model=model,
            batch=batch,
            device=torch.device("cpu"),
        )

    assert metrics.squared_error_sum == 5.0
    assert metrics.absolute_error_sum == 3.0
    assert metrics.target_count == 2
    assert metrics.loss == 2.5
    assert metrics.mae == 1.5


def test_infer_batch_sample_count() -> None:
    """Sample count inference should handle batched, single, and empty inputs."""
    assert (
        infer_batch_sample_count(
            TensorSupervisedBatch(
                input_tensor=torch.empty((3, 2)),
                target_tensor=torch.empty((3, 1)),
                is_batch=True,
            )
        )
        == 3
    )
    assert (
        infer_batch_sample_count(
            TensorSupervisedBatch(
                input_tensor=torch.empty(2),
                target_tensor=torch.empty(1),
                is_batch=False,
            )
        )
        == 1
    )
    assert (
        infer_batch_sample_count(
            TensorSupervisedBatch(
                input_tensor=torch.empty((0, 2)),
                target_tensor=torch.empty((0, 1)),
                is_batch=True,
            )
        )
        == 0
    )
