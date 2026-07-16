"""Dependency-clean supervised tensor batch abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class SupervisedBatch(Protocol):
    """Protocol for supervised tensor batches used by Chipiron trainers."""

    is_batch: bool

    def get_input_layer(self) -> torch.Tensor:
        """Return the model input tensor."""
        ...

    def get_target_value(self) -> torch.Tensor:
        """Return the supervised target tensor."""
        ...


@dataclass(frozen=True, slots=True)
class TensorSupervisedBatch:
    """Concrete supervised batch backed by input and target tensors."""

    input_tensor: torch.Tensor
    target_tensor: torch.Tensor
    is_batch: bool = True
    auxiliary_input_tensors: tuple[torch.Tensor, ...] = ()

    def get_input_layer(self) -> torch.Tensor:
        """Return the model input tensor."""
        return self.input_tensor

    def get_model_input_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return all positional tensors supplied to the model forward call."""
        return (self.input_tensor, *self.auxiliary_input_tensors)

    def get_target_value(self) -> torch.Tensor:
        """Return the supervised target tensor."""
        return self.target_tensor


def move_supervised_batch_to_device(
    batch: SupervisedBatch,
    device: torch.device,
) -> TensorSupervisedBatch:
    """Move one supervised batch to a concrete torch device."""
    if isinstance(batch, TensorSupervisedBatch):
        model_input_tensors = batch.get_model_input_tensors()
    else:
        model_input_tensors = (batch.get_input_layer(),)
    return TensorSupervisedBatch(
        input_tensor=model_input_tensors[0].to(device),
        target_tensor=batch.get_target_value().to(device),
        is_batch=batch.is_batch,
        auxiliary_input_tensors=tuple(
            tensor.to(device) for tensor in model_input_tensors[1:]
        ),
    )
