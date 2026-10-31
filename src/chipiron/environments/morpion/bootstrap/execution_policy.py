"""Explicit bootstrap policy declaration, independent of persisted configuration I/O."""

import os
from typing import TYPE_CHECKING

import torch

from .generation_publication_io import require

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_args import MorpionBootstrapArgs

KEY = "torch_execution_policy"
_STRICT_POLICY: dict[str, object] = {
    "schema_version": 1,
    "profile": "strict_math",
    "intra_op_threads": 1,
    "inter_op_threads": 1,
}


def policy_from_metadata(metadata: Mapping[str, object]) -> dict[str, object] | None:
    """Parse only the supported explicit policy; absence preserves historical bytes."""
    if KEY not in metadata:
        return None
    value = metadata[KEY]
    require(
        isinstance(value, dict)
        and value.keys() == _STRICT_POLICY.keys()
        and all(
            type(value[k]) is type(v) and value[k] == v
            for k, v in _STRICT_POLICY.items()
        ),
        "Unsupported or malformed Torch execution policy.",
    )
    return dict(_STRICT_POLICY)


def policy_metadata(args: MorpionBootstrapArgs) -> dict[str, object]:
    """Keep device and I3 CPU policy separate from process profile/thread policy."""
    if args.execution_profile in (None, "existing"):
        require(
            args.torch_intra_op_threads is None and args.torch_inter_op_threads is None,
            "Process thread options require explicit strict_math.",
        )
        return {}
    require(args.execution_profile == "strict_math", "Unknown execution profile.")
    require(
        isinstance(args.torch_intra_op_threads, int)
        and not isinstance(args.torch_intra_op_threads, bool)
        and isinstance(args.torch_inter_op_threads, int)
        and not isinstance(args.torch_inter_op_threads, bool)
        and args.torch_intra_op_threads == args.torch_inter_op_threads == 1,
        "strict_math requires explicit intra/inter-op 1/1; other counts need qualification.",
    )
    return {KEY: dict(_STRICT_POLICY)}


def validate_supported(args: MorpionBootstrapArgs) -> None:
    """Reject incompatible recipes before filesystem or authoritative mutation."""
    if not policy_metadata(args):
        return
    require(
        args.training_device != "auto"
        and torch.device(args.training_device).type in {"cpu", "cuda"},
        "strict_math requires an explicit CPU or CUDA-namespace device.",
    )
    require(
        os.environ.get("CHIPIRON_CANDIDATE_WORKERS", "1") == "1",
        "strict_math requires serial candidates.",
    )
    require(
        args.continual_learning is None
        or args.continual_learning.transformer_cpu_threads is None,
        "I3 CPU-thread policy plus strict_math requires separate qualification.",
    )
    require(
        torch.get_default_dtype() == torch.float32
        and torch.get_float32_matmul_precision() == "highest",
        "strict_math bootstrap requires existing FP32/highest precision.",
    )

