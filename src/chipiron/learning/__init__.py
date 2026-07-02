"""Common learning/runtime helpers owned by Chipiron."""

from .supervised import (
    SupervisedBatch,
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
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
    "SupervisedBatch",
    "TensorSupervisedBatch",
    "TorchDeviceInfo",
    "TorchDeviceUnavailableError",
    "format_phase_durations",
    "module_device",
    "move_supervised_batch_to_device",
    "parameter_count",
    "resolve_torch_device",
    "state_dict_on_cpu",
    "torch_device_info",
]
