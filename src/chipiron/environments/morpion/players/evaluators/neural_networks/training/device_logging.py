"""Torch device logging for Morpion neural-network training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chipiron.learning.torch_runtime import (
    module_device,
    parameter_count,
    torch_device_info,
)

if TYPE_CHECKING:
    import torch
    from torch import nn

LOGGER = logging.getLogger(__name__)


def log_training_device(
    *,
    model: nn.Module,
    requested_device: str,
    resolved_device: torch.device,
    model_kind: str,
) -> None:
    """Log resolved Torch runtime details for one training evaluator."""
    info = torch_device_info(
        requested_device=requested_device,
        resolved_device=resolved_device,
    )
    LOGGER.info(
        "[train-device] model_kind=%s requested_device=%s resolved_device=%s "
        "model_device=%s cuda_available=%s cuda_device_count=%s "
        "cuda_device_name=%s parameter_count=%s",
        model_kind,
        info.requested_device,
        info.resolved_device,
        module_device(model),
        info.cuda_available,
        info.cuda_device_count,
        info.cuda_device_name,
        parameter_count(model),
    )
