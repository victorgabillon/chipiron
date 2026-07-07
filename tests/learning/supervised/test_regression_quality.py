"""Tests for common supervised regression quality diagnostics."""

from __future__ import annotations

import pytest
import torch

from chipiron.learning.supervised import regression_quality_stats


def test_regression_quality_stats_perfect_prediction() -> None:
    """Perfect predictions should have zero error and ideal quality stats."""
    targets = torch.tensor([1.0, 2.0, 3.0])
    predictions = torch.tensor([1.0, 2.0, 3.0])

    stats = regression_quality_stats(predictions=predictions, targets=targets)

    assert stats.count == 3
    assert stats.mse == pytest.approx(0.0)
    assert stats.mae == pytest.approx(0.0)
    assert stats.pearson_correlation == pytest.approx(1.0)
    assert stats.r2_vs_mean_baseline == pytest.approx(1.0)


def test_regression_quality_stats_mean_predictor() -> None:
    """A constant mean predictor should match the mean baseline."""
    targets = torch.tensor([1.0, 2.0, 3.0])
    predictions = torch.full_like(targets, 2.0)

    stats = regression_quality_stats(predictions=predictions, targets=targets)

    assert stats.mse == pytest.approx(stats.mean_baseline_mse)
    assert stats.r2_vs_mean_baseline == pytest.approx(0.0)
    assert stats.pearson_correlation is None
    assert stats.prediction_std_over_target_std == pytest.approx(0.0)


def test_regression_quality_stats_worse_than_baseline() -> None:
    """Reversed predictions should be anti-correlated and worse than baseline."""
    targets = torch.tensor([1.0, 2.0, 3.0])
    predictions = torch.tensor([3.0, 2.0, 1.0])

    stats = regression_quality_stats(predictions=predictions, targets=targets)

    assert stats.pearson_correlation == pytest.approx(-1.0)
    assert stats.r2_vs_mean_baseline is not None
    assert stats.r2_vs_mean_baseline < 0.0


def test_regression_quality_stats_empty_tensors() -> None:
    """Empty tensors should return empty diagnostics without crashing."""
    stats = regression_quality_stats(
        predictions=torch.empty((0,)),
        targets=torch.empty((0,)),
    )

    assert stats.count == 0
    assert stats.mse is None
    assert stats.pearson_correlation is None


def test_regression_quality_stats_rejects_shape_mismatch() -> None:
    """Prediction and target element counts must match."""
    with pytest.raises(ValueError):
        regression_quality_stats(
            predictions=torch.tensor([1.0, 2.0]),
            targets=torch.tensor([1.0]),
        )
