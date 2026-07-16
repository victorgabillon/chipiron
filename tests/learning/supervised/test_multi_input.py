"""Tests for supervised batches with multiple positional model inputs."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from chipiron.environments.morpion.players.evaluators.neural_networks.training.service import (
    prediction_scale_stats_for_cached_batches,
)
from chipiron.learning.supervised import (
    TensorSupervisedBatch,
    evaluate_regression_batch,
    move_supervised_batch_to_device,
    train_regression_batch,
)


class OneInputRegressor(nn.Module):
    """Small conventional regressor used for compatibility coverage."""

    def __init__(self) -> None:
        """Initialize the one-input projection."""
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict from one feature tensor."""
        return self.linear(features)


class TwoInputRegressor(nn.Module):
    """Small regressor that requires both primary and auxiliary tensors."""

    def __init__(self) -> None:
        """Initialize projections and forward-call observations."""
        super().__init__()
        self.primary_projection = nn.Linear(3, 1)
        self.auxiliary_scale = nn.Parameter(torch.tensor(1.0))
        self.received_primary: torch.Tensor | None = None
        self.received_auxiliary: torch.Tensor | None = None
        self.received_auxiliary_rows: list[torch.Tensor] = []

    def forward(
        self,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        """Predict using both positional inputs."""
        self.received_primary = primary
        self.received_auxiliary = auxiliary
        self.received_auxiliary_rows.append(auxiliary.detach().cpu())
        auxiliary_value = (
            auxiliary.to(dtype=primary.dtype)
            .reshape(primary.shape[0], -1)
            .mean(dim=1, keepdim=True)
        )
        return self.primary_projection(primary) + self.auxiliary_scale * auxiliary_value


def test_single_input_batch_returns_one_model_input() -> None:
    """A conventional batch should expose only its primary input."""
    primary = torch.randn(2, 3)
    target = torch.randn(2, 1)
    batch = TensorSupervisedBatch(
        input_tensor=primary,
        target_tensor=target,
        is_batch=True,
    )

    model_inputs = batch.get_model_input_tensors()

    assert len(model_inputs) == 1
    assert model_inputs[0] is primary
    assert batch.get_input_layer() is primary


def test_multi_input_batch_preserves_input_order() -> None:
    """Primary and auxiliary tensors should retain their positional order."""
    primary = torch.randn(2, 3)
    relations = torch.randint(0, 4, (2, 5, 3), dtype=torch.long)
    metadata = torch.randn(2, 7)
    target = torch.randn(2, 1)
    batch = TensorSupervisedBatch(
        input_tensor=primary,
        target_tensor=target,
        is_batch=True,
        auxiliary_input_tensors=(relations, metadata),
    )

    model_inputs = batch.get_model_input_tensors()

    assert len(model_inputs) == 3
    assert model_inputs[0] is primary
    assert model_inputs[1] is relations
    assert model_inputs[2] is metadata


def test_multi_input_batch_device_transfer_moves_every_tensor() -> None:
    """Batch transfer should preserve all tensor metadata and ordering."""
    primary = torch.randn(2, 3)
    relations = torch.randint(0, 4, (2, 5, 3), dtype=torch.long)
    metadata = torch.randn(2, 7, dtype=torch.float64)
    target = torch.randn(2, 1)
    batch = TensorSupervisedBatch(
        input_tensor=primary,
        target_tensor=target,
        is_batch=False,
        auxiliary_input_tensors=(relations, metadata),
    )

    moved = move_supervised_batch_to_device(batch, torch.device("cpu"))
    model_inputs = moved.get_model_input_tensors()

    assert len(model_inputs) == 3
    assert [tensor.device.type for tensor in model_inputs] == ["cpu", "cpu", "cpu"]
    assert [tensor.shape for tensor in model_inputs] == [
        primary.shape,
        relations.shape,
        metadata.shape,
    ]
    assert [tensor.dtype for tensor in model_inputs] == [
        primary.dtype,
        torch.long,
        torch.float64,
    ]
    assert moved.get_target_value().device.type == "cpu"
    assert moved.get_target_value().shape == target.shape
    assert moved.is_batch is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_multi_input_batch_moves_all_inputs_to_cuda() -> None:
    """CUDA transfer should include integer auxiliary tensors when available."""
    batch = TensorSupervisedBatch(
        input_tensor=torch.randn(2, 3),
        target_tensor=torch.randn(2, 1),
        is_batch=True,
        auxiliary_input_tensors=(
            torch.randint(0, 4, (2, 5, 3), dtype=torch.long),
        ),
    )

    moved = move_supervised_batch_to_device(batch, torch.device("cuda"))
    model_inputs = moved.get_model_input_tensors()

    assert all(tensor.device.type == "cuda" for tensor in model_inputs)
    assert moved.get_target_value().device.type == "cuda"
    assert model_inputs[1].dtype == torch.long


def test_one_input_training_remains_supported() -> None:
    """The generic train helper should retain one-input behavior."""
    model = OneInputRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = TensorSupervisedBatch(
        input_tensor=torch.randn(4, 3),
        target_tensor=torch.randn(4, 1),
        is_batch=True,
    )
    before = model.linear.weight.detach().clone()

    stats = train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        batch=batch,
        device=torch.device("cpu"),
    )

    assert math.isfinite(stats.loss)
    assert stats.sample_count == 4
    assert model.linear.weight.grad is not None
    assert not torch.equal(before, model.linear.weight)


def test_two_input_training_forwards_every_tensor() -> None:
    """The train helper should forward and differentiate through both inputs."""
    model = TwoInputRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _two_input_batch()

    stats = train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        batch=batch,
        device=torch.device("cpu"),
    )

    assert math.isfinite(stats.loss)
    assert stats.sample_count == 4
    assert model.received_primary is not None
    assert model.received_primary.shape == (4, 3)
    assert model.received_primary.device.type == "cpu"
    assert model.received_auxiliary is not None
    assert model.received_auxiliary.shape == (4, 2, 3)
    assert model.received_auxiliary.dtype == torch.long
    assert model.received_auxiliary.device.type == "cpu"
    assert model.primary_projection.weight.grad is not None
    assert torch.isfinite(model.primary_projection.weight.grad).all()
    assert model.auxiliary_scale.grad is not None
    assert torch.isfinite(model.auxiliary_scale.grad)


def test_two_input_evaluation_forwards_every_tensor() -> None:
    """The evaluation helper should use the same multi-input forward path."""
    model = TwoInputRegressor()
    batch = _two_input_batch()
    original_targets = batch.get_target_value().clone()

    model.eval()
    with torch.no_grad():
        metrics = evaluate_regression_batch(
            model=model,
            batch=batch,
            device=torch.device("cpu"),
        )

    assert math.isfinite(metrics.loss)
    assert metrics.target_count == 4
    assert model.received_primary is not None
    assert model.received_primary.shape == (4, 3)
    assert model.received_auxiliary is not None
    assert model.received_auxiliary.shape == (4, 2, 3)
    assert torch.equal(batch.get_target_value(), original_targets)


def test_auxiliary_input_affects_evaluation_prediction() -> None:
    """Evaluation metrics should change when only the auxiliary input changes."""
    model = TwoInputRegressor()
    with torch.no_grad():
        model.primary_projection.weight.zero_()
        model.primary_projection.bias.zero_()
    primary = torch.zeros((2, 3))
    targets = torch.zeros((2, 1))

    with torch.no_grad():
        zero_metrics = evaluate_regression_batch(
            model=model,
            batch=TensorSupervisedBatch(
                input_tensor=primary,
                target_tensor=targets,
                auxiliary_input_tensors=(torch.zeros((2, 2, 3), dtype=torch.long),),
            ),
            device=torch.device("cpu"),
        )
        one_metrics = evaluate_regression_batch(
            model=model,
            batch=TensorSupervisedBatch(
                input_tensor=primary,
                target_tensor=targets,
                auxiliary_input_tensors=(torch.ones((2, 2, 3), dtype=torch.long),),
            ),
            device=torch.device("cpu"),
        )

    assert zero_metrics.squared_error_sum == 0.0
    assert one_metrics.squared_error_sum > zero_metrics.squared_error_sum


def test_cached_prediction_scale_diagnostics_forwards_every_tensor() -> None:
    """Cached scale diagnostics should preserve auxiliary rows and device."""
    model = TwoInputRegressor()
    with torch.no_grad():
        model.primary_projection.weight.zero_()
        model.primary_projection.bias.zero_()
    primary = torch.zeros((4, 3))
    auxiliary = torch.arange(4, dtype=torch.long).reshape(4, 1, 1).expand(-1, 2, 3)
    targets = torch.zeros((4, 1))

    def build_batch(indices: tuple[int, ...]) -> TensorSupervisedBatch:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return TensorSupervisedBatch(
            input_tensor=primary[index_tensor],
            target_tensor=targets[index_tensor],
            auxiliary_input_tensors=(auxiliary[index_tensor],),
        )

    stats = prediction_scale_stats_for_cached_batches(
        model=model,
        batch_builder=build_batch,
        row_count=4,
        batch_size=2,
        device=torch.device("cpu"),
    )

    received_rows = torch.cat(model.received_auxiliary_rows)[:, 0, 0]
    assert torch.equal(received_rows, torch.arange(4))
    assert model.received_primary is not None
    assert model.received_primary.device.type == "cpu"
    assert model.received_auxiliary is not None
    assert model.received_auxiliary.device.type == "cpu"
    assert model.received_auxiliary.dtype == torch.long
    assert stats.count == 4
    assert stats.mean == pytest.approx(1.5)
    assert stats.std is not None and math.isfinite(stats.std)


def _two_input_batch() -> TensorSupervisedBatch:
    """Build one representative two-input regression batch."""
    return TensorSupervisedBatch(
        input_tensor=torch.randn(4, 3),
        target_tensor=torch.randn(4, 1),
        is_batch=True,
        auxiliary_input_tensors=(
            torch.randint(0, 4, (4, 2, 3), dtype=torch.long),
        ),
    )
