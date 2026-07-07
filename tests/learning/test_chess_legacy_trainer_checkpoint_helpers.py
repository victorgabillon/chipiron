"""Tests for legacy chess trainer checkpoint helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


def test_legacy_checkpoint_helpers_import() -> None:
    """The legacy checkpoint helpers should import without chess runtime deps."""
    checkpoint_helpers = _import_legacy_checkpoint_helpers()

    assert checkpoint_helpers.safe_nn_architecture_save is not None
    assert checkpoint_helpers.safe_nn_param_save is not None
    assert checkpoint_helpers.safe_nn_trainer_save is not None


def test_legacy_factory_still_reexports_checkpoint_helpers() -> None:
    """The old factory import path should remain available when deps exist."""
    pytest.importorskip("coral")
    factory = importlib.import_module("chipiron.learningprocesses.nn_trainer.factory")

    assert factory.safe_nn_architecture_save is not None
    assert factory.safe_nn_param_save is not None
    assert factory.safe_nn_trainer_save is not None


def test_safe_nn_param_save_writes_cpu_state_dict(tmp_path: Path) -> None:
    """Parameter checkpoints should be portable from CPU-only runtimes."""
    checkpoint_helpers = _import_legacy_checkpoint_helpers()
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
    checkpoint_helpers = _import_legacy_checkpoint_helpers()
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


def _import_legacy_checkpoint_helpers() -> Any:
    """Import the dependency-light legacy checkpoint helper module."""
    return importlib.import_module(
        "chipiron.learningprocesses.nn_trainer.checkpoint_helpers"
    )
