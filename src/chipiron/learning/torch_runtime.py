"""Small Torch runtime helpers shared by Chipiron learning workflows."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

type TorchExecutionProfile = Literal["existing", "strict_math"]

TORCH_EXECUTION_PROFILE_KEY = "torch_execution_profile"
_PROFILE_LOCK = RLock()
_PROFILE_STACK: list[TorchExecutionProfileInfo] = []


class TorchExecutionProfileError(RuntimeError):
    """An execution profile was invalid, changed, or used outside its scope."""


def configure_torch_process_threads(*, intra_op: int, inter_op: int) -> None:
    """Configure explicit process-lifetime threads before any parallel model work.

    Inter-op cannot be restored/switched after work. An already matching setting
    is left alone; Torch rejects an illegal late change without fallback. This is
    separate from execution profiles and I3's scoped intra-op candidate policy.
    """
    if (
        not isinstance(intra_op, int)
        or isinstance(intra_op, bool)
        or not isinstance(inter_op, int)
        or isinstance(inter_op, bool)
        or min(intra_op, inter_op) < 1
    ):
        message = "Torch process thread counts must be positive integers."
        raise TorchExecutionProfileError(message)
    if torch.get_num_interop_threads() != inter_op:
        torch.set_num_interop_threads(inter_op)
    if torch.get_num_threads() != intra_op:
        torch.set_num_threads(intra_op)


@dataclass(frozen=True, slots=True)
class TorchExecutionProfileInfo:
    """Serializable requested/effective execution settings; no device selection."""

    requested: TorchExecutionProfile
    effective: TorchExecutionProfile
    deterministic: bool
    warn_only: bool
    mha_fastpath: bool
    math_sdp: bool
    flash_sdp: bool
    mem_efficient_sdp: bool
    cudnn_sdp: bool
    torch_version: str
    torch_git_version: str | None
    hip_version: str | None
    cuda_version: str | None
    intra_op_threads: int
    inter_op_threads: int
    default_dtype: str
    float32_matmul_precision: str
    schema_version: int = 1

    def to_metadata(self) -> dict[str, object]:
        """Return an independent JSON-compatible observation."""
        return asdict(self)


def torch_execution_profile_info() -> TorchExecutionProfileInfo:
    """Observe the current scope and backend settings without initializing a GPU."""
    active = _PROFILE_STACK[-1].effective if _PROFILE_STACK else "existing"
    return TorchExecutionProfileInfo(
        requested=active,
        effective=active,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),  # type: ignore[no-untyped-call]
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
        torch_version=str(torch.__version__),
        torch_git_version=torch.version.git_version,
        hip_version=torch.version.hip,
        cuda_version=torch.version.cuda,
        intra_op_threads=torch.get_num_threads(),
        inter_op_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def torch_execution_profile_metadata() -> dict[str, object]:
    """Record opt-in provenance, preserving historical default metadata exactly."""
    info = torch_execution_profile_info()
    if info.effective == "existing":
        return {}
    expected = replace(_PROFILE_STACK[-1], requested=info.requested)
    if info != expected:
        message = "Torch execution settings changed inside the strict_math scope."
        raise TorchExecutionProfileError(message)
    return {TORCH_EXECUTION_PROFILE_KEY: info.to_metadata()}


def validate_torch_execution_profile(metadata: Mapping[str, object]) -> None:
    """Require the same profile and observed runtime for continuation/retry.

    Missing provenance means the historical `existing` profile. Strict runs
    cannot silently inherit those checkpoints or be resumed as historical runs.
    Backend/version/thread changes require a new explicitly qualified lineage.
    """
    current = torch_execution_profile_metadata().get(TORCH_EXECUTION_PROFILE_KEY)
    recorded = metadata.get(TORCH_EXECUTION_PROFILE_KEY)
    matches = (
        TORCH_EXECUTION_PROFILE_KEY not in metadata
        if current is None
        else (
            isinstance(recorded, dict)
            and isinstance(current, dict)
            and recorded.keys() == current.keys()
            and all(
                type(recorded[key]) is type(value) and recorded[key] == value
                for key, value in current.items()
            )
        )
    )
    if not matches:
        message = "Torch execution profile provenance differs from the current scope."
        raise TorchExecutionProfileError(message)


def _restore_execution_profile(
    previous: TorchExecutionProfileInfo, failure: BaseException | None
) -> None:
    """Attempt every restoration without replacing an operation/setup failure."""
    errors: list[BaseException] = []
    for restore in (
        lambda: torch.backends.cuda.enable_math_sdp(previous.math_sdp),
        lambda: torch.backends.cuda.enable_flash_sdp(previous.flash_sdp),
        lambda: torch.backends.cuda.enable_mem_efficient_sdp(
            previous.mem_efficient_sdp
        ),
        lambda: torch.backends.cuda.enable_cudnn_sdp(previous.cudnn_sdp),
        lambda: torch.backends.mha.set_fastpath_enabled(previous.mha_fastpath),
        lambda: torch.use_deterministic_algorithms(
            previous.deterministic, warn_only=previous.warn_only
        ),
    ):
        try:
            restore()
        except BaseException as error:  # pylint: disable=broad-exception-caught
            # Restoration must continue even if one setter fails.
            errors.append(error)
    if errors:
        if failure is not None:
            for restore_error in errors:
                failure.add_note(
                    f"Torch profile restoration also failed: {restore_error!r}"
                )
        else:
            message = "Torch profile restoration failed"
            raise BaseExceptionGroup(message, errors)


@contextmanager
def torch_execution_profile(
    profile: TorchExecutionProfile = "existing",
) -> Iterator[TorchExecutionProfileInfo]:
    """Establish a process profile before starting concurrent model execution.

    The caller owns the whole train/evaluate/resume lifecycle and must join its
    workers before leaving. Spawned processes must enter their own scope. This
    lock rejects competing scope owners, but cannot isolate unmanaged Torch
    calls or other code that changes process-global flags. Same-thread nesting
    is supported; `existing` observes/inherits an enclosing strict scope.

    Strict mode never uses warn_only, changes threads/precision, or catches
    unsupported-operation errors. Only the settings it controls are restored,
    including when setup or the body fails. No implicit profile is installed.
    """
    if profile not in {"existing", "strict_math"}:
        message = f"Unknown Torch execution profile: {profile!r}."
        raise TorchExecutionProfileError(message)
    if not _PROFILE_LOCK.acquire(blocking=False):
        message = "Another thread owns the Torch execution profile scope."
        raise TorchExecutionProfileError(message)
    try:
        previous = torch_execution_profile_info()
        if profile == "existing":
            info = replace(previous, requested=profile)
            _PROFILE_STACK.append(info)
            try:
                yield info
            finally:
                _PROFILE_STACK.pop()
            return
        failure: BaseException | None = None
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.mha.set_fastpath_enabled(False)
            with sdpa_kernel(SDPBackend.MATH):
                info = replace(
                    torch_execution_profile_info(),
                    requested=profile,
                    effective=profile,
                )
                _PROFILE_STACK.append(info)
                try:
                    yield info
                    torch_execution_profile_metadata()
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    _PROFILE_STACK.pop()
        except BaseException as error:
            if failure is None:
                failure = error
                raise
            if failure is error:
                raise
            failure.add_note(f"SDPA scope cleanup also failed: {error!r}")
            # Preserve the model error and its original explicit cause.
            raise failure  # noqa: B904  # pylint: disable=raise-missing-from
        finally:
            _restore_execution_profile(previous, failure)
    finally:
        _PROFILE_LOCK.release()


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
