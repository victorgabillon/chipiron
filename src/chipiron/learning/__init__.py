"""Common learning/runtime helpers owned by Chipiron."""

from .supervised import (
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
from .timing import PhaseDurations, format_phase_durations
from .torch_runtime import (
    TorchDeviceInfo,
    TorchDeviceUnavailableError,
    module_device,
    parameter_count,
    resolve_torch_device,
    state_dict_on_cpu,
    torch_device_info,
)

__all__ = [
    "PhaseDurations",
    "RegressionBatchMetricSums",
    "RegressionBatchTrainStats",
    "RegressionEvaluationStats",
    "RegressionQualityStats",
    "SupervisedBatch",
    "TensorSupervisedBatch",
    "TorchDeviceInfo",
    "TorchDeviceUnavailableError",
    "evaluate_regression_batch",
    "format_phase_durations",
    "infer_batch_sample_count",
    "module_device",
    "move_supervised_batch_to_device",
    "parameter_count",
    "regression_quality_stats",
    "regression_quality_stats_to_metadata",
    "resolve_torch_device",
    "state_dict_on_cpu",
    "torch_device_info",
    "train_regression_batch",
]
