"""Common learning/runtime helpers owned by Chipiron."""

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
    "TorchDeviceInfo",
    "TorchDeviceUnavailableError",
    "format_phase_durations",
    "module_device",
    "parameter_count",
    "resolve_torch_device",
    "state_dict_on_cpu",
    "torch_device_info",
]
