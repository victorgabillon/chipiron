"""Restore the supported legacy chess SGD/StepLR pair onto live parameters."""

import pickle
from typing import TYPE_CHECKING

import torch

from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import (
    get_optimizer_file_path_from,
    get_scheduler_file_path_from,
)

if TYPE_CHECKING:
    from chipiron.utils import MyPath


def restore_sgd_step_lr(
    optimizer: torch.optim.SGD,
    scheduler: torch.optim.lr_scheduler.StepLR,
    folder: MyPath,
) -> None:
    """Load trusted legacy objects as state, preserving the live object graph.

    The chess recipe has one ordered parameter group. Legacy files contain no
    parameter names, so callers must supply the same architecture and parameter
    order. This restores update state, not batch/RNG or scheduling decisions.
    """
    with open(get_optimizer_file_path_from(folder), "rb") as stream:
        saved_optimizer: object = pickle.load(stream)
    with open(get_scheduler_file_path_from(folder), "rb") as stream:
        saved_scheduler: object = pickle.load(stream)
    if (
        not isinstance(saved_optimizer, torch.optim.SGD)
        or saved_optimizer.__class__ is not torch.optim.SGD
    ):
        message = "only SGD optimizer checkpoints are supported"
        raise _restore_error(message)
    if (
        not isinstance(saved_scheduler, torch.optim.lr_scheduler.StepLR)
        or saved_scheduler.__class__ is not torch.optim.lr_scheduler.StepLR
    ):
        message = "only StepLR scheduler checkpoints are supported"
        raise _restore_error(message)
    paired_optimizer = saved_scheduler.optimizer
    if (
        not isinstance(paired_optimizer, torch.optim.SGD)
        or paired_optimizer.__class__ is not torch.optim.SGD
    ):
        message = "the scheduler must contain an SGD optimizer"
        raise _restore_error(message)
    _validate_parameters(optimizer, saved_optimizer, paired_optimizer)
    if saved_scheduler.get_last_lr() != [saved_optimizer.param_groups[0]["lr"]]:
        message = "scheduler and optimizer learning rates differ"
        raise _restore_error(message)

    # Both live objects already exist. Load the scheduler before the optimizer
    # so scheduler initialization can never overwrite a restored learning rate.
    scheduler.load_state_dict(saved_scheduler.state_dict())
    optimizer.load_state_dict(saved_optimizer.state_dict())


def _validate_parameters(
    live: torch.optim.SGD,
    saved: torch.optim.SGD,
    paired: torch.optim.SGD,
) -> None:
    """Reject incompatible layouts and mismatched independently saved objects."""
    if any(len(opt.param_groups) != 1 for opt in (live, saved, paired)):
        message = "only a single parameter group is supported"
        raise _restore_error(message)
    saved_options = {k: v for k, v in saved.param_groups[0].items() if k != "params"}
    paired_options = {k: v for k, v in paired.param_groups[0].items() if k != "params"}
    if saved_options != paired_options:
        message = "optimizer and scheduler checkpoints have different SGD options"
        raise _restore_error(message)
    live_params = live.param_groups[0]["params"]
    saved_params = saved.param_groups[0]["params"]
    paired_params = paired.param_groups[0]["params"]
    if not len(live_params) == len(saved_params) == len(paired_params):
        message = "checkpoint parameter count differs from the live model"
        raise _restore_error(message)
    for current, previous, paired_parameter in zip(
        live_params, saved_params, paired_params, strict=True
    ):
        if current.shape != previous.shape or current.dtype != previous.dtype:
            message = "checkpoint parameter shape or dtype differs from the live model"
            raise _restore_error(message)
        if not torch.equal(previous, paired_parameter):
            message = "optimizer and scheduler checkpoints have different parameters"
            raise _restore_error(message)
        saved_state = saved.state.get(previous, {})
        paired_state = paired.state.get(paired_parameter, {})
        if saved_state.keys() != paired_state.keys() or saved_state.keys() - {
            "momentum_buffer"
        }:
            message = "unsupported or mismatched SGD checkpoint state"
            raise _restore_error(message)
        if "momentum_buffer" in saved_state:
            momentum = saved_state["momentum_buffer"]
            paired_momentum = paired_state["momentum_buffer"]
            if (
                not isinstance(momentum, torch.Tensor)
                or momentum.shape != previous.shape
                or momentum.dtype != previous.dtype
                or not isinstance(paired_momentum, torch.Tensor)
                or not torch.equal(momentum, paired_momentum)
            ):
                message = "incompatible or mismatched checkpoint momentum buffer"
                raise _restore_error(message)


def _restore_error(reason: str) -> ValueError:
    """Describe an unsupported legacy restoration without silently resetting it."""
    return ValueError(f"Cannot restore chess trainer checkpoint: {reason}.")
