"""
Module to create and save neural network trainers and their parameters
"""

import os.path
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.optim as optim
import yaml

from chipiron.learningprocesses.nn_trainer.nn_trainer import NNPytorchTrainer
from chipiron.players.boardevaluators.neural_networks import (
    NeuralNetBoardEvalArgs,
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.factory import (
    get_nn_param_file_path_from,
    get_nn_architecture_file_path_from,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.utils import path
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.dataclass import custom_asdict_factory
from chipiron.utils.small_tools import mkdir_if_not_existing


@dataclass
class NNTrainerArgs:
    """
    Arguments for the NNTrainer class.

    Attributes:
        neural_network (NeuralNetBoardEvalArgs): The arguments for the neural network.
        reuse_existing_trainer (bool): Whether to reuse an existing trainer.
        starting_lr (float): The starting learning rate.
        momentum_op (float): The momentum value.
        scheduler_step_size (int): The step size for the scheduler.
        scheduler_gamma (float): The gamma value for the scheduler.
        saving_intermediate_copy (bool): Whether to save intermediate copies.
    """

    neural_network_folder_path: path = (
        "chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/test_to_keep"
    )
    nn_architecture_file_if_not_reusing_existing_one: path | None = None
    reuse_existing_model: bool = True
    reuse_existing_trainer: bool = False
    starting_lr: float = 0.1
    momentum_op: float = 0.9
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5
    saving_intermediate_copy: bool = True

    def __post_init__(self) -> None:
        if not self.reuse_existing_model:
            if self.nn_architecture_file_if_not_reusing_existing_one is None:
                raise Exception(
                    "Problem because you are asking for a reuse of existing model without specifying an"
                    f" architecture file as we have: reuse_existing_model {self.reuse_existing_model}"
                    f" nn_architecture_file_if_not_reusing_existing_one {self.nn_architecture_file_if_not_reusing_existing_one}"
                )


def get_optimizer_file_path_from(folder_path: path) -> str:
    """
    Returns the file path for the optimizer file in the given folder path.

    Args:
        folder_path (str): The path to the folder containing the optimizer file.

    Returns:
        str: The file path for the optimizer file.

    """
    file_path: str = os.path.join(folder_path, "optimizer.pi")
    return file_path


def get_scheduler_file_path_from(folder_path: path) -> str:
    """Get the file path for the scheduler file in the given folder path.

    Args:
        folder_path (str): The path of the folder containing the scheduler file.

    Returns:
        str: The file path of the scheduler file.
    """
    file_path: str = os.path.join(folder_path, "scheduler.pi")
    return file_path


def get_folder_training_copies_path_from(folder_path: path) -> str:
    """
    Returns the path to the 'training_copies' folder within the given folder path.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        str: The path to the 'training_copies' folder.
    """
    return os.path.join(folder_path, "training_copies")


def create_nn_trainer(args: NNTrainerArgs, nn: ChiNN) -> NNPytorchTrainer:
    """
    Creates an instance of NNPytorchTrainer based on the provided arguments and neural network.

    Args:
        args (NNTrainerArgs): The arguments for the NNTrainer.
        nn (ChiNN): The neural network to be trained.

    Returns:
        NNPytorchTrainer: An instance of NNPytorchTrainer.

    """

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    if args.reuse_existing_trainer:
        file_optimizer_path = get_optimizer_file_path_from(
            folder_path=args.neural_network_folder_path
        )
        with open(file_optimizer_path, "rb") as file_optimizer:
            optimizer = pickle.load(file_optimizer)

        file_scheduler_path = get_scheduler_file_path_from(
            folder_path=args.neural_network_folder_path
        )
        with open(file_scheduler_path, "rb") as file_scheduler:
            scheduler = pickle.load(file_scheduler)

    else:
        optimizer = optim.SGD(
            nn.parameters(),
            lr=args.starting_lr,
            momentum=args.momentum_op,
            weight_decay=0.000,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )

    if args.saving_intermediate_copy:
        folder_path_training_copies = get_folder_training_copies_path_from(
            args.neural_network_folder_path
        )
        mkdir_if_not_existing(folder_path_training_copies)

    return NNPytorchTrainer(net=nn, optimizer=optimizer, scheduler=scheduler)


def safe_nn_architecture_save(
    nn_architecture_args: NeuralNetArchitectureArgs, nn_param_folder_name: path
) -> None:
    """ """
    path_to_param_file = get_nn_architecture_file_path_from(nn_param_folder_name)
    try:
        print(f"saving architecture to file: {path_to_param_file}")
        with open(path_to_param_file, "w") as file_architecture:
            yaml.dump(
                asdict(
                    nn_architecture_args,
                    dict_factory=custom_asdict_factory,
                ),
                file_architecture,
                default_flow_style=False,
            )
    except KeyboardInterrupt:
        exit(-1)


def safe_nn_param_save(
    nn: ChiNN, nn_param_folder_name: path, training_copy: bool = False
) -> None:
    """
    Save the parameters of a neural network to a file.

    Args:
        nn (ChiNN): The neural network to save.
        training_copy (bool, optional): Whether to save a training copy of the parameters. Defaults to False.
    """
    folder_path = nn_param_folder_name
    folder_path_training_copies = get_folder_training_copies_path_from(folder_path)
    nn_file_path = get_nn_param_file_path_from(folder_path)
    path_to_param_file: path
    if training_copy:
        now = datetime.now()  # current date and time
        path_to_param_file = os.path.join(
            folder_path_training_copies, now.strftime("%A-%m-%d-%Y--%H:%M:%S:%f")
        )
    else:
        path_to_param_file = nn_file_path
    try:
        print(f"saving to file: {path_to_param_file}")
        with open(path_to_param_file, "wb") as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
        with open(path_to_param_file + "_save", "wb") as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
    except KeyboardInterrupt:
        with open(path_to_param_file + "_save", "wb") as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
        exit(-1)


def safe_nn_trainer_save(nn_trainer: NNPytorchTrainer, nn_folder_path: path) -> None:
    """
    Safely saves the optimizer and scheduler of the given NNPytorchTrainer object to files.

    Args:
        nn_trainer (NNPytorchTrainer): The NNPytorchTrainer object containing the optimizer and scheduler to be saved.
        args (NeuralNetBoardEvalArgs): The arguments specifying the folder paths and other parameters.

    Returns:
        None
    """
    file_optimizer_path = get_optimizer_file_path_from(nn_folder_path)
    file_scheduler_path = get_scheduler_file_path_from(nn_folder_path)
    try:
        with open(file_optimizer_path, "wb") as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path, "wb") as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
        with open(str(file_optimizer_path) + "_save", "wb") as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path + "_save", "wb") as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
    except KeyboardInterrupt:
        with open(file_optimizer_path + "_save", "wb") as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path + "_save", "wb") as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
        exit(-1)
