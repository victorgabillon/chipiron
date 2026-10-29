"""Tensor forwarding and gradient updates, independent of training recipes."""

from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from torch import Tensor, nn
    from torch.optim import Optimizer


def forward_tensors(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Forward positional tensors without changing device, gradients or model mode."""
    return cast("Tensor", model(*inputs))


def backward_and_step(
    loss: Tensor,
    optimizer: Optimizer,
    *,
    backward_phase: AbstractContextManager[None] | None = None,
    optimizer_phase: AbstractContextManager[None] | None = None,
) -> None:
    """Backpropagate once and step once, with optional caller-owned timing scopes.

    The caller owns gradient clearing, loss/finite policy, scheduling and metrics.
    No scalar extraction or device synchronization is performed here. Existing
    gradients accumulate; even a zero loss still reaches the optimizer.
    """
    with backward_phase if backward_phase is not None else nullcontext():
        backward = cast("Callable[[], None]", loss.backward)
        backward()
    with optimizer_phase if optimizer_phase is not None else nullcontext():
        optimizer.step()
