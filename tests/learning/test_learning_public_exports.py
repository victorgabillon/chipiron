"""Smoke tests for stable public learning exports."""

from __future__ import annotations

from chipiron.learning import (
    RegressionBatchMetricSums,
    RegressionBatchTrainStats,
    RegressionEvaluationStats,
    RegressionQualityStats,
    SupervisedBatch,
    TensorSupervisedBatch,
    evaluate_regression_batch,
    infer_batch_sample_count,
    move_supervised_batch_to_device,
    regression_quality_stats,
    regression_quality_stats_to_metadata,
    train_regression_batch,
)
from chipiron.learning.supervised import (
    RegressionQualityStats as SupervisedRegressionQualityStats,
)
from chipiron.learning.supervised import (
    regression_quality_stats as supervised_regression_quality_stats,
)
from chipiron.learning.supervised import (
    regression_quality_stats_to_metadata as supervised_regression_quality_stats_to_metadata,
)


def test_learning_public_exports_are_available() -> None:
    """Common learning exports should be importable from stable locations."""
    exported_objects = (
        RegressionBatchMetricSums,
        RegressionBatchTrainStats,
        RegressionEvaluationStats,
        RegressionQualityStats,
        SupervisedBatch,
        TensorSupervisedBatch,
        evaluate_regression_batch,
        infer_batch_sample_count,
        move_supervised_batch_to_device,
        regression_quality_stats,
        regression_quality_stats_to_metadata,
        train_regression_batch,
    )

    assert all(exported_object is not None for exported_object in exported_objects)
    assert SupervisedRegressionQualityStats is RegressionQualityStats
    assert supervised_regression_quality_stats is regression_quality_stats
    assert (
        supervised_regression_quality_stats_to_metadata
        is regression_quality_stats_to_metadata
    )
