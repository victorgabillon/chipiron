"""Operational training timing; independent of numerical execution profiles."""

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, TypedDict

from .torch_runtime import synchronize_torch_device_if_needed

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch


type TrainingTimingMode = Literal["synchronized", "host_enqueue"]


def validate_training_timing_mode(mode: str) -> TrainingTimingMode:
    """Reject unknown modes without silently changing historical timing semantics."""
    if mode == "synchronized":
        return "synchronized"
    if mode == "host_enqueue":
        return "host_enqueue"
    raise ValueError("Training timing mode must be synchronized or host_enqueue.")  # noqa: TRY003


class TrainingTimingOptions(TypedDict, total=False):
    """Only operational timing may be forwarded to an existing service recipe."""

    timing_mode: TrainingTimingMode


def bootstrap_training_timing_options() -> TrainingTimingOptions:
    """Capture an explicit operational environment option for a fresh job only."""
    mode = validate_training_timing_mode(
        os.environ.get("CHIPIRON_TRAINING_TIMING_MODE", "synchronized")
    )
    return {} if mode == "synchronized" else {"timing_mode": mode}


def synchronize_training_phases(mode: TrainingTimingMode, device: torch.device) -> bool:
    """Resolve fine timers; opt-in supports CPU controls and CUDA/ROCm only."""
    validate_training_timing_mode(mode)
    if mode == "host_enqueue" and device.type not in {"cpu", "cuda"}:
        message = "host_enqueue training timing supports CPU and CUDA/ROCm only."
        raise ValueError(message)
    return mode == "synchronized"


@contextmanager
def device_complete_scope(device: torch.device) -> Iterator[None]:
    """Keep enclosing service timing device-complete, including exceptional exit."""
    synchronize_torch_device_if_needed(device)
    try:
        yield
    finally:
        synchronize_torch_device_if_needed(device)
