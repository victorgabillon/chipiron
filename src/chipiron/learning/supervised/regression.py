"""Generic supervised regression batch mechanics."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from chipiron.learning.supervised.batches import (
    SupervisedBatch,
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from chipiron.learning.timing import PhaseDurations


@dataclass(frozen=True, slots=True)
class RegressionBatchTrainStats:
    """Stats returned by one supervised regression train batch."""

    loss: float
    squared_error_sum: float
    absolute_error_sum: float
    sample_count: int
    target_count: int


@dataclass(frozen=True, slots=True)
class RegressionBatchMetricSums:
    """Accumulated regression metric sums for one evaluated batch."""

    squared_error_sum: float
    absolute_error_sum: float
    target_count: int

    @property
    def loss(self) -> float:
        """Return MSE over all targets, or 0.0 for empty batches."""
        if self.target_count == 0:
            return 0.0
        return self.squared_error_sum / self.target_count

    @property
    def mae(self) -> float:
        """Return MAE over all targets, or 0.0 for empty batches."""
        if self.target_count == 0:
            return 0.0
        return self.absolute_error_sum / self.target_count


@dataclass(frozen=True, slots=True)
class RegressionEvaluationStats:
    """Final regression evaluation statistics."""

    loss: float
    mae: float
    sample_count: int
    target_count: int
    phase_durations: dict[str, float]
    elapsed_seconds: float


def synchronize_torch_device_if_needed(device: torch.device) -> None:
    """Synchronize pending device work when required for accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_torch_phase(
    timings: PhaseDurations,
    phase: str,
    device: torch.device,
    *,
    synchronize_cuda: bool = True,
) -> Iterator[None]:
    """Measure a Torch phase, optionally synchronizing CUDA around it."""
    if synchronize_cuda:
        synchronize_torch_device_if_needed(device)
    started_at = perf_counter()
    try:
        yield
    finally:
        if synchronize_cuda:
            synchronize_torch_device_if_needed(device)
        timings.add_duration(phase, perf_counter() - started_at)


def infer_batch_sample_count(batch: SupervisedBatch) -> int:
    """Infer number of samples represented by a supervised batch."""
    input_tensor = batch.get_input_layer()
    if batch.is_batch and input_tensor.ndim > 0:
        return int(input_tensor.shape[0])
    return 1


def _forward_supervised_batch(
    model: nn.Module,
    batch: TensorSupervisedBatch,
) -> torch.Tensor:
    """Run a model with every positional input tensor from one batch."""
    return cast("torch.Tensor", model(*batch.get_model_input_tensors()))


def train_regression_batch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: SupervisedBatch,
    device: torch.device,
    timings: PhaseDurations | None = None,
    synchronize_cuda: bool = True,
) -> RegressionBatchTrainStats:
    """Run one supervised regression optimization step."""
    phase = _phase_timer(timings, device, synchronize_cuda=synchronize_cuda)
    sample_count = infer_batch_sample_count(batch)

    with phase("batch_transfer"):
        device_batch = move_supervised_batch_to_device(batch, device)
    with phase("zero_grad"):
        optimizer.zero_grad()
    with phase("forward"):
        predictions = _forward_supervised_batch(model, device_batch)
    targets = device_batch.get_target_value()
    with phase("loss"):
        loss = criterion(predictions, targets)
    with phase("backward"):
        loss.backward()
    with phase("optimizer_step"):
        optimizer.step()
    with phase("metric_accumulation"):
        errors = predictions.detach() - targets
        squared_error_sum = float(torch.sum(errors * errors).item())
        absolute_error_sum = float(torch.sum(torch.abs(errors)).item())
        target_count = int(targets.numel())
        loss_value = float(loss.detach().cpu().item())

    return RegressionBatchTrainStats(
        loss=loss_value,
        squared_error_sum=squared_error_sum,
        absolute_error_sum=absolute_error_sum,
        sample_count=sample_count,
        target_count=target_count,
    )


def evaluate_regression_batch(
    *,
    model: nn.Module,
    batch: SupervisedBatch,
    device: torch.device,
    timings: PhaseDurations | None = None,
    synchronize_cuda: bool = True,
) -> RegressionBatchMetricSums:
    """Evaluate one supervised regression batch and return metric sums."""
    phase = _phase_timer(timings, device, synchronize_cuda=synchronize_cuda)

    with phase("batch_transfer"):
        device_batch = move_supervised_batch_to_device(batch, device)
    with phase("forward"):
        predictions = _forward_supervised_batch(model, device_batch)
    targets = device_batch.get_target_value()
    with phase("metric_accumulation"):
        errors = predictions - targets
        squared_error_sum = float(torch.sum(errors * errors).item())
        absolute_error_sum = float(torch.sum(torch.abs(errors)).item())
        target_count = int(targets.numel())

    return RegressionBatchMetricSums(
        squared_error_sum=squared_error_sum,
        absolute_error_sum=absolute_error_sum,
        target_count=target_count,
    )


def _phase_timer(
    timings: PhaseDurations | None,
    device: torch.device,
    *,
    synchronize_cuda: bool,
) -> _PhaseTimer:
    """Return a phase timing callable for optional instrumentation."""

    def phase(name: str) -> AbstractContextManager[None]:
        if timings is None:
            return nullcontext()
        return timed_torch_phase(
            timings,
            name,
            device,
            synchronize_cuda=synchronize_cuda,
        )

    return phase


type _PhaseTimer = Callable[[str], AbstractContextManager[None]]
