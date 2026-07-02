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

    def get_input_layer(self) -> torch.Tensor:
        """Return the model input tensor."""
        return self.input_tensor

    def get_target_value(self) -> torch.Tensor:
        """Return the supervised target tensor."""
        return self.target_tensor


def move_supervised_batch_to_device(
    batch: SupervisedBatch,
    device: torch.device,
) -> TensorSupervisedBatch:
    """Move one supervised batch to a concrete torch device."""
    return TensorSupervisedBatch(
        input_tensor=batch.get_input_layer().to(device),
        target_tensor=batch.get_target_value().to(device),
        is_batch=batch.is_batch,
    )
