"""Tests for shared Torch runtime helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chipiron.learning.torch_runtime import (
    TorchDeviceUnavailableError,
    module_device,
    parameter_count,
    resolve_torch_device,
    state_dict_on_cpu,
)


def test_resolve_torch_device_cpu() -> None:
    """Explicit CPU requests should resolve to CPU."""
    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_resolve_torch_device_auto_uses_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should fall back to CPU without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")


def test_resolve_torch_device_auto_uses_cuda_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should choose CUDA when CUDA is reported available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_torch_device("auto") == torch.device("cuda")


def test_resolve_torch_device_cuda_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA requests should fail clearly without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(TorchDeviceUnavailableError):
        resolve_torch_device("cuda")


def test_resolve_torch_device_cuda_index_must_be_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA indices should be checked without allocating CUDA tensors."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_torch_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(TorchDeviceUnavailableError):
        resolve_torch_device("cuda:999")


@pytest.mark.parametrize("requested_device", ("", "   "))
def test_resolve_torch_device_rejects_blank_strings(requested_device: str) -> None:
    """Blank device strings should be rejected."""
    with pytest.raises(TorchDeviceUnavailableError):
        resolve_torch_device(requested_device)


def test_parameter_count_counts_linear_weight_and_bias() -> None:
    """Parameter counting should include weights and bias tensors."""
    assert parameter_count(nn.Linear(3, 2)) == 3 * 2 + 2


def test_module_device_returns_cpu_by_default() -> None:
    """Fresh CPU modules should report CPU."""
    assert module_device(nn.Linear(3, 2)) == torch.device("cpu")


def test_state_dict_on_cpu_returns_cpu_tensors() -> None:
    """State dict copies should be detached CPU tensors."""
    state_dict = state_dict_on_cpu(nn.Linear(3, 2))

    assert state_dict
    assert all(tensor.device == torch.device("cpu") for tensor in state_dict.values())
