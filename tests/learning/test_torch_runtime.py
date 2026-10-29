"""Tests for shared Torch runtime helpers."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from chipiron.learning.torch_runtime import (
    TorchDeviceUnavailableError,
    module_device,
    parameter_count,
    resolve_torch_device,
    state_dict_on_cpu,
    synchronize_torch_device_if_needed,
    torch_device_info,
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


@pytest.mark.parametrize("backend", ["mps", "xpu"])
def test_explicit_accelerator_availability_and_indices(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Dispatch contracts only: mocked availability is not training validation."""
    availability_api = torch.backends.mps if backend == "mps" else torch.xpu
    device_api = torch.mps if backend == "mps" else torch.xpu
    monkeypatch.setattr(availability_api, "is_available", lambda: False)
    for request in (backend, f"{backend}:0"):
        with pytest.raises(
            TorchDeviceUnavailableError, match=f"{backend.upper()} is unavailable"
        ):
            resolve_torch_device(request)
    monkeypatch.setattr(availability_api, "is_available", lambda: True)
    monkeypatch.setattr(device_api, "device_count", lambda: 1)
    assert resolve_torch_device(backend) == torch.device(backend)
    assert resolve_torch_device(f"{backend}:0") == torch.device(f"{backend}:0")
    with pytest.raises(
        TorchDeviceUnavailableError, match="outside the available range"
    ):
        resolve_torch_device(f"{backend}:1")


@pytest.mark.parametrize("cuda_available", [False, True])
def test_auto_does_not_probe_other_accelerators(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool
) -> None:
    """Auto retains the established CUDA-namespace/CPU scientific default."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    unexpected = Mock(side_effect=AssertionError("Auto must not probe MPS/XPU"))
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected)
    monkeypatch.setattr(torch.xpu, "is_available", unexpected)
    assert resolve_torch_device(" AUTO ").type == ("cuda" if cuda_available else "cpu")
    unexpected.assert_not_called()


@pytest.mark.parametrize("device_request", ["not-a-device", "cuda:-1", "xpu:wrong"])
def test_invalid_request_has_shared_error(device_request: str) -> None:
    """Malformed strings fail at the shared boundary."""
    with pytest.raises(
        TorchDeviceUnavailableError, match="Invalid torch device request"
    ):
        resolve_torch_device(device_request)


@pytest.mark.parametrize(
    ("cuda_version", "hip_version", "backend"),
    [("12.8", None, "nvidia_cuda"), (None, "7.14.60850", "amd_rocm")],
)
def test_diagnostics_distinguish_cuda_namespace_backends(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    hip_version: str | None,
    backend: str,
) -> None:
    """Build identity distinguishes vendors; device names are not heuristics."""
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    get_name = Mock(return_value="device name")
    monkeypatch.setattr(torch.cuda, "get_device_name", get_name)
    device = resolve_torch_device(" CUDA:1 ")
    info = torch_device_info(requested_device=" CUDA:1 ", resolved_device=device)
    assert info.backend == backend
    assert info.resolved_device == "cuda:1"
    assert info.cuda_version == cuda_version
    assert info.hip_version == hip_version
    assert info.cuda_available and info.cuda_device_count == 2
    assert info.cuda_device_name == "device name"
    get_name.assert_called_once_with(device)
    cpu_info = torch_device_info(
        requested_device="cpu", resolved_device=torch.device("cpu")
    )
    assert cpu_info.backend == "cpu"
    assert cpu_info.cuda_device_name is None


def test_diagnostics_survive_failed_name_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """An optional name query must not prevent diagnostics."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", Mock(side_effect=RuntimeError("query failed"))
    )
    assert (
        torch_device_info(
            requested_device="cuda", resolved_device=torch.device("cuda")
        ).cuda_device_name
        is None
    )


@pytest.mark.parametrize("name", ["cpu", "cuda:1", "mps:0", "xpu:1"])
def test_synchronization_uses_pinned_backend_signatures(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Mocks check routing and argument signatures, never accelerator execution."""
    calls = {backend: Mock() for backend in ("cuda", "mps", "xpu")}
    for backend, call in calls.items():
        monkeypatch.setattr(getattr(torch, backend), "synchronize", call)
    device = torch.device(name)
    synchronize_torch_device_if_needed(device)
    for backend, call in calls.items():
        if device.type == backend:
            if backend == "mps":
                call.assert_called_once_with()
            else:
                call.assert_called_once_with(device)
        else:
            call.assert_not_called()


def test_synchronization_failure_is_not_silenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed accelerator waits cannot be presented as successful timings."""
    monkeypatch.setattr(
        torch.xpu, "synchronize", Mock(side_effect=RuntimeError("device failure"))
    )
    with pytest.raises(RuntimeError, match="device failure"):
        synchronize_torch_device_if_needed(torch.device("xpu"))


def test_legacy_sync_imports_share_one_implementation() -> None:
    """Existing import paths retain the shared runtime callable."""
    from chipiron import learning
    from chipiron.learning import supervised
    from chipiron.learning.supervised import regression

    assert (
        learning.synchronize_torch_device_if_needed
        is synchronize_torch_device_if_needed
    )
    assert (
        supervised.synchronize_torch_device_if_needed
        is synchronize_torch_device_if_needed
    )
    assert (
        regression.synchronize_torch_device_if_needed
        is synchronize_torch_device_if_needed
    )


@pytest.mark.parametrize("backend", ["cuda", "mps", "xpu"])
@pytest.mark.parametrize("enabled", [True, False])
def test_legacy_timing_flag_controls_accelerator_waits(
    monkeypatch: pytest.MonkeyPatch, backend: str, enabled: bool
) -> None:
    """Retain opt-out and both timing boundaries without executing GPU work."""
    from chipiron.learning.supervised.regression import timed_torch_phase
    from chipiron.learning.timing import PhaseDurations

    wait = Mock()
    monkeypatch.setattr(getattr(torch, backend), "synchronize", wait)
    timings = PhaseDurations()
    with timed_torch_phase(
        timings, "forward", torch.device(backend), synchronize_cuda=enabled
    ):
        assert wait.call_count == int(enabled)
    assert wait.call_count == 2 * int(enabled)
    assert "forward" in timings.as_dict()
