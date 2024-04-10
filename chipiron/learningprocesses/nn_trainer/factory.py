""" 
Module to create and save neural network trainers and their parameters
"""
import pickle
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.optim as optim

from chipiron.learningprocesses.nn_trainer.nn_trainer import NNPytorchTrainer
from chipiron.players.boardevaluators.neural_networks import NeuralNetBoardEvalArgs
from chipiron.players.boardevaluators.neural_networks.factory import get_folder_path_from, get_nn_param_file_path_from
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.small_tools import mkdir


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

    neural_network: NeuralNetBoardEvalArgs = field(
        default_factory=lambda: NeuralNetBoardEvalArgs(
            nn_type='pp2d2_2_leaky',
            nn_param_folder_name='foo'
        )
    )
    reuse_existing_trainer: bool = False
    starting_lr: float = .1
    momentum_op: float = .9
    scheduler_step_size: int = 1
    scheduler_gamma: float = .5
    saving_intermediate_copy: bool = True


def get_optimizer_file_path_from(folder_path: str) -> str:
    """
    Returns the file path for the optimizer file in the given folder path.

    Args:
        folder_path (str): The path to the folder containing the optimizer file.

    Returns:
        str: The file path for the optimizer file.

    """
    file_path = folder_path + '/optimizer.pi'
    return file_path


def get_scheduler_file_path_from(folder_path: str) -> str:
    """Get the file path for the scheduler file in the given folder path.

    Args:
        folder_path (str): The path of the folder containing the scheduler file.

    Returns:
        str: The file path of the scheduler file.
    """
    file_path = folder_path + '/scheduler.pi'
    return file_path


def get_folder_training_copies_path_from(folder_path: str) -> str:
    """
    Returns the path to the 'training_copies' folder within the given folder path.
    
    Parameters:
        folder_path (str): The path to the folder.
        
    Returns:
        str: The path to the 'training_copies' folder.
    """
    return folder_path + '/training_copies'


def create_nn_trainer(
        args: NNTrainerArgs,
        nn: ChiNN
) -> NNPytorchTrainer:
    """
    Creates an instance of NNPytorchTrainer based on the provided arguments and neural network.

    Args:
        args (NNTrainerArgs): The arguments for the NNTrainer.
        nn (ChiNN): The neural network to be trained.

    Returns:
        NNPytorchTrainer: An instance of NNPytorchTrainer.

    """
    args_nn = args.neural_network
    nn_folder_path = get_folder_path_from(
        nn_type=args_nn.nn_type,
        nn_param_folder_name=args_nn.nn_param_folder_name
    )

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    if args.reuse_existing_trainer:
        file_optimizer_path = get_optimizer_file_path_from(nn_folder_path)
        with open(file_optimizer_path, 'rb') as file_optimizer:
            optimizer = pickle.load(file_optimizer)

        file_scheduler_path = get_scheduler_file_path_from(nn_folder_path)
        with open(file_scheduler_path, 'rb') as file_scheduler:
            scheduler = pickle.load(file_scheduler)

    else:
        optimizer = optim.SGD(
            nn.parameters(),
            lr=args.starting_lr,
            momentum=args.momentum_op,
            weight_decay=.000)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )

    if args.saving_intermediate_copy:
        folder_path_training_copies = get_folder_training_copies_path_from(nn_folder_path)
        mkdir(folder_path_training_copies)

    return NNPytorchTrainer(
        net=nn,
        optimizer=optimizer,
        scheduler=scheduler
    )


def safe_nn_param_save(
        nn: ChiNN,
        args: NeuralNetBoardEvalArgs,
        training_copy: bool = False
) -> None:
    """
    Save the parameters of a neural network to a file.

    Args:
        nn (ChiNN): The neural network to save.
        args (NeuralNetBoardEvalArgs): The arguments containing information about the neural network.
        training_copy (bool, optional): Whether to save a training copy of the parameters. Defaults to False.
    """
    folder_path = get_folder_path_from(args.nn_type, args.nn_param_folder_name)
    folder_path_training_copies = get_folder_training_copies_path_from(folder_path)
    nn_file_path = get_nn_param_file_path_from(folder_path)
    if training_copy:
        now = datetime.now()  # current date and time
        path_to_param_file = folder_path_training_copies + '/' + now.strftime("%A-%m-%d-%Y--%H:%M:%S:%f")
    else:
        path_to_param_file = nn_file_path
    try:
        with open(path_to_param_file, 'wb') as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
        with open(path_to_param_file + '_save', 'wb') as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
    except KeyboardInterrupt:
        with open(path_to_param_file + '_save', 'wb') as fileNNW:
            torch.save(nn.state_dict(), fileNNW)
        exit(-1)


def safe_nn_trainer_save(
        nn_trainer: NNPytorchTrainer,
        args: NeuralNetBoardEvalArgs
) -> None:
    """
    Safely saves the optimizer and scheduler of the given NNPytorchTrainer object to files.

    Args:
        nn_trainer (NNPytorchTrainer): The NNPytorchTrainer object containing the optimizer and scheduler to be saved.
        args (NeuralNetBoardEvalArgs): The arguments specifying the folder paths and other parameters.

    Returns:
        None
    """
    nn_folder_path = get_folder_path_from(args.nn_type, args.nn_param_folder_name)
    file_optimizer_path = get_optimizer_file_path_from(nn_folder_path)
    file_scheduler_path = get_scheduler_file_path_from(nn_folder_path)
    try:
        with open(file_optimizer_path, 'wb') as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path, 'wb') as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
        with open(file_optimizer_path + '_save', 'wb') as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path + '_save', 'wb') as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
    except KeyboardInterrupt:
        with open(file_optimizer_path + '_save', 'wb') as file_optimizer:
            pickle.dump(nn_trainer.optimizer, file_optimizer)
        with open(file_scheduler_path + '_save', 'wb') as file_scheduler:
            pickle.dump(nn_trainer.scheduler, file_scheduler)
        exit(-1)
