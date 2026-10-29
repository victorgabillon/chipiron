"""Real legacy checkpoint restoration onto a new live chess model."""

import pickle
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import safe_nn_trainer_save
from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
    _create_optimizer_and_scheduler,
)
from chipiron.scripts.learn_nn_supervised.training_args import SupervisedTrainingArgs

if TYPE_CHECKING:
    from pathlib import Path

type TrainingState = tuple[
    nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]


def _model() -> nn.Linear:
    model = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.4]]))
        model.bias.fill_(0.1)
    return model


def _args(*, reuse: bool = False) -> SupervisedTrainingArgs:
    return SupervisedTrainingArgs(
        starting_lr=0.07,
        momentum_op=0.8,
        scheduler_step_size=2,
        scheduler_gamma=0.3,
        reuse_existing_trainer=reuse,
        saving_intermediate_copy=False,
    )


def _update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    index: int,
    *,
    event: bool,
) -> None:
    inputs = torch.tensor([[1.0, -0.5], [0.3, 0.8]], dtype=torch.float64)
    targets = torch.tensor([[0.4], [-0.2]], dtype=torch.float64) + index / 20
    optimizer.zero_grad()
    nn.L1Loss()(model(inputs), targets).backward()
    optimizer.step()
    if event:
        scheduler.step()


def _assert_training_equal(left: TrainingState, right: TrainingState) -> None:
    left_model, left_optimizer, left_scheduler = left
    right_model, right_optimizer, right_scheduler = right
    for lhs, rhs in zip(left_model.parameters(), right_model.parameters(), strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
        left_state = left_optimizer.state.get(lhs, {})
        right_state = right_optimizer.state.get(rhs, {})
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            torch.testing.assert_close(
                left_state[key], right_state[key], rtol=0, atol=0
            )
    assert (
        left_optimizer.state_dict()["param_groups"]
        == right_optimizer.state_dict()["param_groups"]
    )
    assert left_scheduler.state_dict() == right_scheduler.state_dict()
    assert left_scheduler.get_last_lr() == right_scheduler.get_last_lr()


@pytest.mark.parametrize("split", [0, 1, 2, 3, 4])
def test_resume_matches_updates_across_scheduler_decay(
    tmp_path: Path, split: int
) -> None:
    """Saved momentum and explicit scheduler events continue exactly on CPU."""
    model = _model()
    optimizer, scheduler = _create_optimizer_and_scheduler(
        args=_args(), model=model, saving_folder=tmp_path
    )
    events = [True, False, True, True, False, True, True]
    for index in range(split):
        _update(model, optimizer, scheduler, index, event=events[index])
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    safe_nn_trainer_save(
        SimpleNamespace(optimizer=optimizer, scheduler=scheduler), tmp_path
    )

    resumed = _model()
    resumed.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    # Saved recipe/state wins over fresh initialization settings.
    args = _args(reuse=True)
    args.starting_lr, args.momentum_op = 0.9, 0.0
    args.scheduler_step_size, args.scheduler_gamma = 99, 0.9
    restored_optimizer, restored_scheduler = _create_optimizer_and_scheduler(
        args=args, model=resumed, saving_folder=tmp_path
    )
    assert all(
        saved is live
        for saved, live in zip(
            restored_optimizer.param_groups[0]["params"],
            resumed.parameters(),
            strict=True,
        )
    )
    assert restored_scheduler.optimizer is restored_optimizer
    reference = (model, optimizer, scheduler)
    continuation = (resumed, restored_optimizer, restored_scheduler)
    _assert_training_equal(reference, continuation)
    for index in range(split, len(events)):
        _update(*reference, index, event=events[index])
        _update(*continuation, index, event=events[index])
        _assert_training_equal(reference, continuation)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.07 * 0.3**2)


def test_fresh_initialization(tmp_path: Path) -> None:
    """Fresh training retains SGD/StepLR configuration and copy-directory setup."""
    model = _model()
    args = _args()
    args.saving_intermediate_copy = True
    optimizer, scheduler = _create_optimizer_and_scheduler(
        args=args, model=model, saving_folder=tmp_path
    )
    assert type(optimizer) is torch.optim.SGD
    assert type(scheduler) is torch.optim.lr_scheduler.StepLR
    assert scheduler.optimizer is optimizer
    assert optimizer.param_groups[0]["params"] == list(model.parameters())
    assert optimizer.param_groups[0]["lr"] == 0.07
    assert optimizer.param_groups[0]["momentum"] == 0.8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert scheduler.step_size == 2
    assert scheduler.gamma == 0.3
    assert (tmp_path / "training_copies").is_dir()


def test_legacy_independent_pickles(tmp_path: Path) -> None:
    """Original optimizer.pi/scheduler.pi objects remain supported."""
    model = _model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.6, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.4)
    _update(model, optimizer, scheduler, 0, event=True)
    for name, value in (("optimizer", optimizer), ("scheduler", scheduler)):
        with (tmp_path / f"{name}.pi").open("wb") as stream:
            pickle.dump(value, stream)
    resumed = _model()
    resumed.load_state_dict(model.state_dict())
    restored = _create_optimizer_and_scheduler(
        args=_args(reuse=True), model=resumed, saving_folder=tmp_path
    )
    assert restored[1].optimizer is restored[0]
    _update(model, optimizer, scheduler, 1, event=True)
    _update(resumed, *restored, 1, event=True)
    _assert_training_equal((model, optimizer, scheduler), (resumed, *restored))


@pytest.mark.parametrize(
    "invalid",
    ["optimizer", "scheduler", "groups", "shape", "count", "momentum", "pair"],
)
def test_incompatible_checkpoint_rejected(tmp_path: Path, invalid: str) -> None:
    """Unsupported recipes and incompatible model/state layouts fail explicitly."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.6)
    if invalid == "optimizer":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    if invalid == "groups":
        optimizer = torch.optim.SGD(
            [{"params": [model.weight]}, {"params": [model.bias]}], lr=0.03
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    if invalid == "scheduler":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    _update(model, optimizer, scheduler, 0, event=True)
    if invalid == "momentum":
        optimizer.state[model.weight]["momentum_buffer"] = torch.ones(9)
    if invalid == "pair":
        other = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
        scheduler = torch.optim.lr_scheduler.StepLR(other, step_size=2)
    safe_nn_trainer_save(
        SimpleNamespace(optimizer=optimizer, scheduler=scheduler), tmp_path
    )
    resumed = nn.Linear(3, 1) if invalid == "shape" else _model()
    if invalid == "count":
        resumed = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"[Cc]heckpoint|[Rr]estor|[Ss]upported"):
        _create_optimizer_and_scheduler(
            args=_args(reuse=True), model=resumed, saving_folder=tmp_path
        )
