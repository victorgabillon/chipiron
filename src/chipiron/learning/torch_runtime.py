"""Small Torch runtime helpers shared by Chipiron learning workflows."""

from dataclasses import dataclass

import torch
from torch import nn


class TorchDeviceUnavailableError(RuntimeError):
    """Raised when a requested torch device cannot be used."""

    @classmethod
    def empty_request(cls) -> TorchDeviceUnavailableError:
        """Return the error for an empty device request."""
        return cls("Torch device must be a non-empty string.")

    @classmethod
    def invalid_request(
        cls,
        requested_device: str,
    ) -> TorchDeviceUnavailableError:
        """Return the error for a device string Torch cannot parse."""
        return cls(f"Invalid torch device request: {requested_device!r}.")

    @classmethod
    def cuda_unavailable(
        cls,
        requested_device: str,
    ) -> TorchDeviceUnavailableError:
        """Return the error for an unavailable CUDA-namespace device."""
        return cls(
            f"Requested torch device {requested_device!r}, but the CUDA namespace "
            "is unavailable (NVIDIA CUDA or AMD ROCm, depending on the Torch build)."
        )

    @classmethod
    def cuda_index_out_of_range(
        cls,
        *,
        requested_device: str,
        device_index: int,
        cuda_device_count: int,
    ) -> TorchDeviceUnavailableError:
        """Return the error for an unavailable explicit CUDA index."""
        return cls(
            f"Requested torch device {requested_device!r}, but CUDA device index "
            f"{device_index} is outside the available range [0, {cuda_device_count})."
        )


@dataclass(frozen=True, slots=True)
class TorchDeviceInfo:
    """Resolved torch-device information for logging and metadata."""

    requested_device: str
    resolved_device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    backend: str = "unknown"
    cuda_version: str | None = None
    hip_version: str | None = None


def resolve_torch_device(requested_device: str = "auto") -> torch.device:
    """Resolve a user-facing device request into a concrete torch.device.

    ``"auto"`` chooses the CUDA namespace (NVIDIA CUDA or AMD ROCm) when
    available, otherwise CPU. It never selects MPS or XPU automatically.
    Explicit CUDA, MPS and XPU requests require availability and valid indices;
    availability does not establish operator support or training qualification.
    """
    if not isinstance(requested_device, str) or not requested_device.strip():
        raise TorchDeviceUnavailableError.empty_request()

    normalized_device = requested_device.strip().lower()
    if normalized_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")

    try:
        device = torch.device(normalized_device)
    except (RuntimeError, TypeError) as exc:
        raise TorchDeviceUnavailableError.invalid_request(requested_device) from exc

    if device.type == "cuda":
        _validate_cuda_device(
            device_index=device.index, requested_device=requested_device
        )
    elif device.type in {"mps", "xpu"}:
        _validate_optional_accelerator(
            device_type=device.type,
            device_index=device.index,
            requested_device=requested_device,
        )
    return device


def synchronize_torch_device_if_needed(device: torch.device) -> None:
    """Wait for device work using backend APIs present in pinned Torch 2.14.1.

    CPU is a no-op. ROCm uses the CUDA API; MPS synchronization is device-wide
    and takes no device argument. Errors propagate instead of hiding bad timings.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def torch_device_info(
    *,
    requested_device: str,
    resolved_device: torch.device,
) -> TorchDeviceInfo:
    """Return stable device information for logs and persisted metadata."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    cuda_device_name = None
    if resolved_device.type == "cuda" and cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(resolved_device)
        except AssertionError, RuntimeError:
            cuda_device_name = None
    return TorchDeviceInfo(
        requested_device=requested_device,
        resolved_device=str(resolved_device),
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        backend=(
            ("amd_rocm" if torch.version.hip is not None else "nvidia_cuda")
            if resolved_device.type == "cuda"
            else resolved_device.type
        ),
        cuda_version=torch.version.cuda,
        hip_version=torch.version.hip,
    )


def module_device(module: nn.Module) -> torch.device:
    """Return the device of the first module parameter or CPU for parameterless modules."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and non-trainable parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def state_dict_on_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU copy of the module state dict."""
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def _validate_cuda_device(
    *,
    device_index: int | None,
    requested_device: str,
) -> None:
    """Validate CUDA availability and index bounds for one concrete device."""
    if not torch.cuda.is_available():
        raise TorchDeviceUnavailableError.cuda_unavailable(requested_device)
    cuda_device_count = torch.cuda.device_count()
    if device_index is not None and not 0 <= device_index < cuda_device_count:
        raise TorchDeviceUnavailableError.cuda_index_out_of_range(
            requested_device=requested_device,
            device_index=device_index,
            cuda_device_count=cuda_device_count,
        )


def _validate_optional_accelerator(
    *, device_type: str, device_index: int | None, requested_device: str
) -> None:
    """Check explicit MPS/XPU requests without qualifying training support."""
    available = (
        torch.backends.mps.is_available()
        if device_type == "mps"
        else torch.xpu.is_available()
    )
    if not available:
        message = (
            f"Requested torch device {requested_device!r}, but "
            f"{device_type.upper()} is unavailable in this Torch build or host."
        )
        raise TorchDeviceUnavailableError(message)
    count = (
        torch.mps.device_count() if device_type == "mps" else torch.xpu.device_count()
    )
    if device_index is not None and not 0 <= device_index < count:
        message = (
            f"Requested torch device {requested_device!r}, but "
            f"{device_type.upper()} device index {device_index} is outside "
            f"the available range [0, {count})."
        )
        raise TorchDeviceUnavailableError(message)
