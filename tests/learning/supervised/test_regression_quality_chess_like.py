"""Regression quality smoke tests for chess-like value targets."""

from __future__ import annotations

import torch

from chipiron.learning.supervised import regression_quality_stats


def test_regression_quality_accepts_chess_like_value_targets() -> None:
    """Common quality diagnostics should handle value-evaluator target ranges."""
    targets = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    predictions = torch.tensor([-0.8, -0.4, 0.1, 0.4, 0.9])

    stats = regression_quality_stats(predictions=predictions, targets=targets)

    assert stats.count == 5
    assert stats.mse is not None
    assert stats.mae is not None
    assert stats.pearson_correlation is not None
    assert stats.pearson_correlation > 0.9
    assert stats.r2_vs_mean_baseline is not None
    assert stats.r2_vs_mean_baseline > 0.0
