"""Scalar criterion reconstruction shared by both chess script paths."""

import pytest
import torch
from torch import nn

from chipiron.learning.supervised import (
    RegressionBatchMetricSums,
    scalar_loss_from_regression_metrics,
)


@pytest.mark.parametrize("loss_type", [nn.L1Loss, nn.MSELoss])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_scalar_reconstruction_matches_multi_output_criterion(
    loss_type: type[nn.L1Loss] | type[nn.MSELoss], reduction: str
) -> None:
    """Mean uses the target count, including every output of each sample."""
    predictions = torch.tensor([[0.0, 2.0, 4.0], [3.0, -1.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, -2.0]])
    errors = predictions - targets
    metrics = RegressionBatchMetricSums(
        squared_error_sum=float(errors.square().sum()),
        absolute_error_sum=float(errors.abs().sum()),
        target_count=targets.numel(),
    )
    criterion = loss_type(reduction=reduction)
    assert scalar_loss_from_regression_metrics(
        criterion=criterion, metrics=metrics
    ) == pytest.approx(float(criterion(predictions, targets)))
    assert (
        scalar_loss_from_regression_metrics(
            criterion=criterion, metrics=RegressionBatchMetricSums(0.0, 0.0, 0)
        )
        == 0.0
    )


@pytest.mark.parametrize("loss_type", [nn.L1Loss, nn.MSELoss])
@pytest.mark.parametrize("reduction", ["none", "median", ""])
def test_unsupported_scalar_reductions_fail(
    loss_type: type[nn.L1Loss] | type[nn.MSELoss], reduction: str
) -> None:
    """Aggregate sums cannot reconstruct unreduced or unknown criteria."""
    with pytest.raises(TypeError):
        scalar_loss_from_regression_metrics(
            criterion=loss_type(reduction=reduction),
            metrics=RegressionBatchMetricSums(5.0, 3.0, 2),
        )


def test_unsupported_scalar_criterion_fails() -> None:
    """Do not silently approximate a different loss from L1/MSE sums."""
    with pytest.raises(TypeError):
        scalar_loss_from_regression_metrics(
            criterion=nn.SmoothL1Loss(), metrics=RegressionBatchMetricSums(5.0, 3.0, 2)
        )


def test_chess_scripts_share_scalar_reconstruction() -> None:
    """Retain both script helper names while routing them to one implementation."""
    from chipiron.scripts.evaluate_models.evaluate_models import (
        _scalar_loss_from_regression_metrics as evaluate_scalar,
    )
    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        _scalar_loss_from_regression_metrics as learn_scalar,
    )

    assert evaluate_scalar is learn_scalar is scalar_loss_from_regression_metrics
