"""Checkpoint helpers for supervised chess neural-network training."""

from __future__ import annotations

import os.path
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, cast

import torch
import yaml
from torch import nn

from chipiron.learning import state_dict_on_cpu
from chipiron.utils.dataclass import custom_asdict_factory
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from chipiron.utils import MyPath


class ReadableWeightsModule(Protocol):
    """Model protocol for legacy readable-weight logging."""

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """Log model weights in the legacy readable YAML format."""


class OptimizerSchedulerHolder(Protocol):
    """Protocol for objects exposing legacy trainer checkpoint state."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def get_optimizer_file_path_from(folder_path: MyPath) -> str:
    """Return the optimizer checkpoint path in a trainer folder."""
    return os.path.join(folder_path, "optimizer.pi")


def get_scheduler_file_path_from(folder_path: MyPath) -> str:
    """Return the scheduler checkpoint path in a trainer folder."""
    return os.path.join(folder_path, "scheduler.pi")


def get_folder_training_copies_path_from(folder_path: MyPath) -> str:
    """Return the folder path for legacy intermediate training copies."""
    return os.path.join(folder_path, "training_copies")


def _get_nn_param_file_path_from(
    folder_path: MyPath, file_name: str | None = None
) -> tuple[str, str]:
    """Return legacy parameter and readable-weight paths."""
    if file_name is None:
        nn_param_file_path = os.path.join(folder_path, "param")
    else:
        nn_param_file_path = os.path.join(folder_path, file_name)
    return nn_param_file_path + ".pt", nn_param_file_path + ".yaml"


def _get_nn_architecture_file_path_from(folder_path: MyPath) -> str:
    """Return the legacy architecture YAML path."""
    return os.path.join(folder_path, "architecture.yaml")


def safe_nn_architecture_save(
    nn_architecture_args: object, nn_param_folder_name: MyPath
) -> None:
    """Save the architecture of a neural network to a legacy YAML file."""
    path_to_param_file = _get_nn_architecture_file_path_from(nn_param_folder_name)
    try:
        chipiron_logger.info("saving architecture to file: %s", path_to_param_file)
        with open(path_to_param_file, "w", encoding="utf-8") as file_architecture:
            yaml.dump(
                asdict(
                    nn_architecture_args,
                    dict_factory=custom_asdict_factory,
                ),
                file_architecture,
                default_flow_style=False,
            )
    except KeyboardInterrupt:
        sys.exit(-1)


def safe_nn_param_save(
    nn_module: nn.Module,
    nn_param_folder_name: MyPath,
    file_name: str | None = None,
    training_copy: bool = False,
) -> None:
    """Save portable CPU-normalized neural-network parameters."""
    folder_path = nn_param_folder_name
    folder_path_training_copies = get_folder_training_copies_path_from(folder_path)

    nn_file_path_pt: str
    file_name_yaml: str
    nn_file_path_pt, file_name_yaml = _get_nn_param_file_path_from(
        folder_path=folder_path, file_name=file_name
    )
    path_to_param_file: MyPath
    if training_copy:
        now = datetime.now()
        path_to_param_file = os.path.join(
            folder_path_training_copies, now.strftime("%A-%m-%d-%Y--%H:%M:%S:%f")
        )
    else:
        path_to_param_file = nn_file_path_pt
    try:
        chipiron_logger.info("saving to file: %s", path_to_param_file)
        state_dict = state_dict_on_cpu(nn_module)
        with open(path_to_param_file, "wb") as file_nnw:
            torch.save(state_dict, file_nnw)
            cast("ReadableWeightsModule", nn_module).log_readable_model_weights_to_file(
                file_path=file_name_yaml
            )
        with open(path_to_param_file + "_save", "wb") as file_nnw:
            torch.save(state_dict, file_nnw)
    except KeyboardInterrupt:
        state_dict = state_dict_on_cpu(nn_module)
        with open(path_to_param_file + "_save", "wb") as file_nnw:
            torch.save(state_dict, file_nnw)
        sys.exit(-1)


def safe_nn_trainer_save(
    training_state: OptimizerSchedulerHolder,
    nn_folder_path: MyPath,
) -> None:
    """Safely save optimizer and scheduler state in the legacy trainer format."""
    file_optimizer_path = get_optimizer_file_path_from(nn_folder_path)
    file_scheduler_path = get_scheduler_file_path_from(nn_folder_path)
    try:
        with open(file_optimizer_path, "wb") as file_optimizer:
            pickle.dump(training_state.optimizer, file_optimizer)
        with open(file_scheduler_path, "wb") as file_scheduler:
            pickle.dump(training_state.scheduler, file_scheduler)
        with open(str(file_optimizer_path) + "_save", "wb") as file_optimizer:
            pickle.dump(training_state.optimizer, file_optimizer)
        with open(file_scheduler_path + "_save", "wb") as file_scheduler:
            pickle.dump(training_state.scheduler, file_scheduler)
    except KeyboardInterrupt:
        with open(file_optimizer_path + "_save", "wb") as file_optimizer:
            pickle.dump(training_state.optimizer, file_optimizer)
        with open(file_scheduler_path + "_save", "wb") as file_scheduler:
            pickle.dump(training_state.scheduler, file_scheduler)
        sys.exit(-1)
