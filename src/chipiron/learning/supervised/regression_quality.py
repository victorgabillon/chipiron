"""Common supervised regression quality diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch


@dataclass(frozen=True, slots=True)
class RegressionQualityStats:
    """Prediction/target quality diagnostics for scalar regression."""

    count: int
    target_mean: float | None
    target_std: float | None
    target_min: float | None
    target_max: float | None
    prediction_mean: float | None
    prediction_std: float | None
    prediction_min: float | None
    prediction_max: float | None
    residual_mean: float | None
    residual_std: float | None
    residual_min: float | None
    residual_max: float | None
    mse: float | None
    mae: float | None
    mean_baseline_mse: float | None
    zero_baseline_mse: float | None
    r2_vs_mean_baseline: float | None
    pearson_correlation: float | None
    prediction_std_over_target_std: float | None


def regression_quality_stats(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> RegressionQualityStats:
    """Return scalar regression quality diagnostics for predictions and targets."""
    flattened_predictions = predictions.detach().float().cpu().reshape(-1)
    flattened_targets = targets.detach().float().cpu().reshape(-1)
    count = int(flattened_targets.numel())
    if int(flattened_predictions.numel()) != count:
        raise _prediction_target_count_mismatch_error()
    if count == 0:
        return RegressionQualityStats(
            count=0,
            target_mean=None,
            target_std=None,
            target_min=None,
            target_max=None,
            prediction_mean=None,
            prediction_std=None,
            prediction_min=None,
            prediction_max=None,
            residual_mean=None,
            residual_std=None,
            residual_min=None,
            residual_max=None,
            mse=None,
            mae=None,
            mean_baseline_mse=None,
            zero_baseline_mse=None,
            r2_vs_mean_baseline=None,
            pearson_correlation=None,
            prediction_std_over_target_std=None,
        )

    target_mean_tensor = torch.mean(flattened_targets)
    prediction_mean_tensor = torch.mean(flattened_predictions)
    target_centered = flattened_targets - target_mean_tensor
    prediction_centered = flattened_predictions - prediction_mean_tensor
    residuals = flattened_predictions - flattened_targets
    target_std = float(torch.std(flattened_targets, unbiased=False).item())
    prediction_std = float(torch.std(flattened_predictions, unbiased=False).item())
    mean_baseline_mse = float(torch.mean(target_centered * target_centered).item())
    mse = float(torch.mean(residuals * residuals).item())
    pearson_correlation: float | None = None
    if target_std > 0.0 and prediction_std > 0.0:
        covariance = float(torch.mean(target_centered * prediction_centered).item())
        pearson_correlation = covariance / (target_std * prediction_std)
    r2_vs_mean_baseline = (
        None if mean_baseline_mse <= 0.0 else 1.0 - (mse / mean_baseline_mse)
    )
    prediction_std_over_target_std = (
        None if target_std <= 0.0 else prediction_std / target_std
    )
    return RegressionQualityStats(
        count=count,
        target_mean=float(target_mean_tensor.item()),
        target_std=target_std,
        target_min=float(torch.min(flattened_targets).item()),
        target_max=float(torch.max(flattened_targets).item()),
        prediction_mean=float(prediction_mean_tensor.item()),
        prediction_std=prediction_std,
        prediction_min=float(torch.min(flattened_predictions).item()),
        prediction_max=float(torch.max(flattened_predictions).item()),
        residual_mean=float(torch.mean(residuals).item()),
        residual_std=float(torch.std(residuals, unbiased=False).item()),
        residual_min=float(torch.min(residuals).item()),
        residual_max=float(torch.max(residuals).item()),
        mse=mse,
        mae=float(torch.mean(torch.abs(residuals)).item()),
        mean_baseline_mse=mean_baseline_mse,
        zero_baseline_mse=float(
            torch.mean(flattened_targets * flattened_targets).item()
        ),
        r2_vs_mean_baseline=r2_vs_mean_baseline,
        pearson_correlation=pearson_correlation,
        prediction_std_over_target_std=prediction_std_over_target_std,
    )


def regression_quality_stats_to_metadata(
    stats: RegressionQualityStats,
) -> dict[str, object]:
    """Return a JSON-friendly mapping for regression quality stats."""
    return cast("dict[str, object]", asdict(stats))


def _prediction_target_count_mismatch_error() -> ValueError:
    """Return a clean error for incompatible prediction/target tensors."""
    return ValueError("predictions and targets must contain the same number of values.")


__all__ = [
    "RegressionQualityStats",
    "regression_quality_stats",
    "regression_quality_stats_to_metadata",
]
