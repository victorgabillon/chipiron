"""
Module for creating neural networks and neural network board evaluators.
"""

import os.path
import sys
from typing import Any

from chipiron.players.boardevaluators.neural_networks import NeuralNetBoardEvalArgs
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    BoardToInput,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_364_bti import (
    RepresentationBTI,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.RepresentationType import (
    RepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.models.nn_pp1 import NetPP1
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2 import NetPP2
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2 import NetPP2D2
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2 import (
    NetPP2D2_2,
)
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2leaky import (
    NetPP2D2_2_LEAKY,
)
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2prelu import (
    NetPP2D2_2_PRELU,
)
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2rrelu import (
    NetPP2D2_2_RRELU,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import (
    OneDToValueWhite,
    OutputValueConverter,
)
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.small_tools import mkdir


def get_folder_path_from(
        nn_type: str,
        nn_param_folder_name: str
) -> str:
    """
    Get the folder path for the neural network parameters.

    Args:
        nn_type (str): The type of neural network.
        nn_param_folder_name (str): The folder name for the neural network parameters.

    Returns:
        str: The folder path for the neural network parameters.
    """
    print('nn_type', nn_type)
    folder_path = os.path.join('data/players/board_evaluators/nn_pytorch/nn_' + nn_type, nn_param_folder_name)
    return folder_path


def get_nn_param_file_path_from(
        folder_path: str
) -> str:
    """
    Get the file path for the neural network parameters.

    Args:
        folder_path (str): The folder path for the neural network parameters.

    Returns:
        str: The file path for the neural network parameters.
    """
    nn_param_file_path: str = os.path.join(folder_path, 'param.pt')
    return nn_param_file_path


def create_nn(
        args: NeuralNetBoardEvalArgs,
        create_file: bool = False
) -> ChiNN:
    """
    Create a neural network.

    Args:
        args (NeuralNetBoardEvalArgs): The arguments for creating the neural network.
        create_file (bool, optional): Whether to create the parameter file if it doesn't exist. Defaults to False.

    Returns:
        ChiNN: The created neural network.
    """
    folder_path = get_folder_path_from(
        nn_type=args.nn_type,
        nn_param_folder_name=args.nn_param_folder_name
    )
    mkdir(folder_path)

    net: ChiNN
    match args.nn_type:
        case 'pp1':
            net = NetPP1()
        case 'pp2':
            net = NetPP2()
        case 'pp2d2':
            net = NetPP2D2()
        case 'pp2d2_2':
            net = NetPP2D2_2()
        case 'pp2d2_2_leaky':
            net = NetPP2D2_2_LEAKY()
        case 'pp2d2_2_rrelu':
            net = NetPP2D2_2_RRELU()
        case 'pp2d2_2_prelu':
            net = NetPP2D2_2_PRELU()
        case other:
            sys.exit(f'Create NN: can not find {other} in file {__name__}')

    nn_param_file_path = get_nn_param_file_path_from(folder_path)
    print('nn_param_file_path', nn_param_file_path, create_file)
    net.load_from_file_or_init_weights(nn_param_file_path, create_file)

    net.eval()
    return net


def create_nn_board_eval(
        arg: NeuralNetBoardEvalArgs,
        representation_type: RepresentationType,
        create_file: bool = False,
) -> NNBoardEvaluator:
    """
    Create a neural network board evaluator.

    Args:
        arg (NeuralNetBoardEvalArgs): The arguments for creating the neural network board evaluator.
        create_file (bool, optional): Whether to create the parameter file if it doesn't exist. Defaults to False.

    Returns:
        NNBoardEvaluator: The created neural network board evaluator.
    """
    net = create_nn(arg, create_file=create_file)
    output_and_value_converter: OutputValueConverter = OneDToValueWhite(point_of_view=net.evaluation_point_of_view)
    representation_factory: RepresentationFactory[Any] | None = create_board_representation_factory(
        board_representation_factory_type=representation_type
    )
    assert (representation_factory is not None)
    board_to_input_converter: BoardToInput = RepresentationBTI(
        representation_factory=representation_factory
    )
    return NNBoardEvaluator(
        net=net,
        output_and_value_converter=output_and_value_converter,
        board_to_input_converter=board_to_input_converter
    )
