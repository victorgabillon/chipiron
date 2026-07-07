"""Tests for supervised chess checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chipiron.scripts.learn_nn_supervised import checkpoint_helpers


def test_checkpoint_helpers_import() -> None:
    """The canonical checkpoint helpers should import without chess runtime deps."""
    assert checkpoint_helpers.safe_nn_architecture_save is not None
    assert checkpoint_helpers.safe_nn_param_save is not None
    assert checkpoint_helpers.safe_nn_trainer_save is not None


def test_safe_nn_param_save_writes_cpu_state_dict(tmp_path: Path) -> None:
    """Parameter checkpoints should be portable from CPU-only runtimes."""
    model = _ReadableLinear(3, 1)

    if torch.cuda.is_available():
        model = model.to("cuda")

    checkpoint_helpers.safe_nn_param_save(model, tmp_path)

    state_dict = torch.load(tmp_path / "param.pt", map_location="cpu")

    assert state_dict
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
    assert (tmp_path / "param.pt_save").exists()
    assert (tmp_path / "param.yaml").read_text(encoding="utf-8") == "readable\n"


def test_safe_nn_trainer_save_accepts_optimizer_scheduler_holder(
    tmp_path: Path,
) -> None:
    """Trainer checkpoint saving should only require optimizer and scheduler."""
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    holder = _OptimizerSchedulerHolder(optimizer=optimizer, scheduler=scheduler)

    checkpoint_helpers.safe_nn_trainer_save(holder, tmp_path)

    assert (tmp_path / "optimizer.pi").exists()
    assert (tmp_path / "scheduler.pi").exists()
    assert (tmp_path / "optimizer.pi_save").exists()
    assert (tmp_path / "scheduler.pi_save").exists()


@dataclass(frozen=True, slots=True)
class _OptimizerSchedulerHolder:
    """Small holder matching the checkpoint helper protocol."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


class _ReadableLinear(nn.Linear):
    """Small model exposing the legacy readable-weights hook."""

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """Write a tiny readable weights marker."""
        Path(file_path).write_text("readable\n", encoding="utf-8")
