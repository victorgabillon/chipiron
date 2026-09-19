"""Tests for common supervised tensor batch helpers."""

from __future__ import annotations

import torch

from chipiron.learning.supervised import (
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
)


def test_tensor_supervised_batch_returns_input_tensor() -> None:
    """TensorSupervisedBatch should expose the input tensor unchanged."""
    input_tensor = torch.tensor([[1.0, 2.0]])
    target_tensor = torch.tensor([[3.0]])
    batch = TensorSupervisedBatch(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
    )

    assert batch.get_input_layer() is input_tensor


def test_tensor_supervised_batch_returns_target_tensor() -> None:
    """TensorSupervisedBatch should expose the target tensor unchanged."""
    input_tensor = torch.tensor([[1.0, 2.0]])
    target_tensor = torch.tensor([[3.0]])
    batch = TensorSupervisedBatch(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
    )

    assert batch.get_target_value() is target_tensor


def test_tensor_supervised_batch_preserves_batch_flag() -> None:
    """TensorSupervisedBatch should preserve its explicit batch flag."""
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([1.0, 2.0]),
        target_tensor=torch.tensor([3.0]),
        is_batch=False,
    )

    assert batch.is_batch is False


def test_move_supervised_batch_to_cpu_returns_cpu_tensors() -> None:
    """Moving a supervised batch to CPU should return CPU tensors."""
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0]]),
        target_tensor=torch.tensor([[3.0]]),
    )

    moved = move_supervised_batch_to_device(batch, torch.device("cpu"))

    assert moved.get_input_layer().device == torch.device("cpu")
    assert moved.get_target_value().device == torch.device("cpu")


def test_move_supervised_batch_to_cpu_preserves_batch_flag() -> None:
    """Moving a supervised batch should preserve the batch flag."""
    batch = TensorSupervisedBatch(
        input_tensor=torch.tensor([[1.0, 2.0]]),
        target_tensor=torch.tensor([[3.0]]),
        is_batch=True,
    )

    moved = move_supervised_batch_to_device(batch, torch.device("cpu"))

    assert moved.is_batch is True
