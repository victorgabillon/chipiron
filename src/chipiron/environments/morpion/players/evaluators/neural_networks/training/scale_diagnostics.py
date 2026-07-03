"""Target and prediction scale diagnostics for Morpion neural training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch


@dataclass(frozen=True, slots=True)
class TensorScaleStats:
    """Scalar distribution statistics for one tensor."""

    count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    abs_mean: float | None
    abs_max: float | None


def tensor_scale_stats(tensor: torch.Tensor) -> TensorScaleStats:
    """Return scalar distribution stats for one tensor."""
    flattened = tensor.detach().float().cpu().reshape(-1)
    count = int(flattened.numel())
    if count == 0:
        return TensorScaleStats(
            count=0,
            mean=None,
            std=None,
            min=None,
            max=None,
            abs_mean=None,
            abs_max=None,
        )
    absolute_values = torch.abs(flattened)
    return TensorScaleStats(
        count=count,
        mean=float(torch.mean(flattened).item()),
        std=float(torch.std(flattened, unbiased=False).item()),
        min=float(torch.min(flattened).item()),
        max=float(torch.max(flattened).item()),
        abs_mean=float(torch.mean(absolute_values).item()),
        abs_max=float(torch.max(absolute_values).item()),
    )


def tensor_scale_stats_to_metadata(stats: TensorScaleStats) -> dict[str, object]:
    """Return a JSON-friendly mapping for tensor scale stats."""
    return cast("dict[str, object]", asdict(stats))


def zero_prediction_mse(target_tensor: torch.Tensor) -> float | None:
    """Return MSE for zero predictions against one target tensor."""
    flattened = target_tensor.detach().float().cpu().reshape(-1)
    if flattened.numel() == 0:
        return None
    return float(torch.mean(flattened * flattened).item())


def target_scale_metadata(target_tensor: torch.Tensor) -> dict[str, object]:
    """Return target scale metadata including zero-prediction MSE."""
    stats = tensor_scale_stats(target_tensor)
    metadata = tensor_scale_stats_to_metadata(stats)
    metadata["zero_prediction_mse"] = zero_prediction_mse(target_tensor)
    return metadata


__all__ = [
    "TensorScaleStats",
    "target_scale_metadata",
    "tensor_scale_stats",
    "tensor_scale_stats_to_metadata",
    "zero_prediction_mse",
]
