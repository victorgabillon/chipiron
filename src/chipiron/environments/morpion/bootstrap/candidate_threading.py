"""Scoped, identity-bearing CPU threading for the serial transformer candidate."""

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from .candidate_resources import candidate_cpu_budget, require_serial_cpu
from .generation_publication_io import require

if TYPE_CHECKING:
    from collections.abc import Iterator

KEY = "cpu_threading"
LOGGER = logging.getLogger(__name__)


def validate_thread_provenance(metadata: dict[str, Any], requested: int | None) -> None:
    """Reject missing, changed or silently disabled accepted thread settings."""
    evidence = metadata.get(KEY)
    if requested is None:
        require(evidence is None, "Accepted candidate CPU thread setting changed.")
        return
    require(
        isinstance(evidence, dict)
        and isinstance(evidence.get("requested_intra_op_threads"), int)
        and not isinstance(evidence.get("requested_intra_op_threads"), bool)
        and isinstance(evidence.get("actual_intra_op_threads"), int)
        and not isinstance(evidence.get("actual_intra_op_threads"), bool)
        and evidence["requested_intra_op_threads"] == requested
        and evidence["actual_intra_op_threads"] == requested,
        "Accepted candidate CPU thread provenance differs from policy.",
    )


@contextmanager
def candidate_threads(
    requested: int | None, *, name: str, device: str
) -> Iterator[dict[str, Any]]:
    """Change only intra-op threads in Training; restore even on BaseException."""
    if requested is None:
        yield {}
        return
    require(
        name == "transformer_v1"
        and isinstance(requested, int)
        and not isinstance(requested, bool)
        and requested in {1, 2},
        "Scoped CPU threads support only transformer_v1 with 1 or 2 threads.",
    )
    require_serial_cpu(device)
    capacity, reserved, budget = candidate_cpu_budget()
    require(
        budget >= requested,
        "Transformer CPU budget is insufficient after pipeline reserves.",
    )
    previous = torch.get_num_threads()
    observation = {
        "requested_intra_op_threads": requested,
        "previous_intra_op_threads": previous,
        "interop_threads": torch.get_num_interop_threads(),
        "cpu_capacity": capacity,
        "reserved_cpus": reserved,
        "cpu_budget": budget,
    }
    try:
        started = time.perf_counter()
        torch.set_num_threads(requested)
        observation["switch_in_s"] = time.perf_counter() - started
        observation["actual_intra_op_threads"] = torch.get_num_threads()
        require(
            observation["actual_intra_op_threads"] == requested,
            "Torch did not apply requested transformer CPU threads.",
        )
        LOGGER.info("[train-stream] transformer_cpu_threading=%s", observation)
        yield observation
        require(
            torch.get_num_threads() == requested,
            "Transformer CPU threads changed during training.",
        )
    finally:
        started = time.perf_counter()
        torch.set_num_threads(previous)
        observation["restore_s"] = time.perf_counter() - started
        observation["restored_intra_op_threads"] = torch.get_num_threads()
        require(
            observation["restored_intra_op_threads"] == previous,
            "Torch did not restore Training worker CPU threads.",
        )
