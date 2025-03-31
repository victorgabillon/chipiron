"""
Module for creating neural networks and neural network board evaluators.
"""

import os.path
import sys
from enum import Enum
from typing import Any

import dacite

from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    BoardToInputFunction,
    create_board_to_input,
)
from chipiron.players.boardevaluators.neural_networks.models.nn_p1 import NetP1
from chipiron.players.boardevaluators.neural_networks.models.nn_p2 import NetP2
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
from chipiron.players.boardevaluators.neural_networks.models.tranformer_one import (
    TransformerOne,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.factory import (
    create_output_converter,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import (
    OutputValueConverter,
)
from chipiron.utils import path, yaml_fetch_args_in_file
from chipiron.utils.chi_nn import ChiNN


def get_nn_param_file_path_from(folder_path: path) -> str:
    """
    Get the file path for the neural network parameters.

    Args:
        folder_path (str): The folder path for the neural network parameters.

    Returns:
        str: The file path for the neural network parameters.
    """
    nn_param_file_path: str = os.path.join(folder_path, "param.pt")
    return nn_param_file_path


def get_nn_architecture_file_path_from(folder_path: path) -> str:
    """
    Get the file path for the architecture parameters.

    Args:
        folder_path (str): The folder path for the architecture parameters.

    Returns:
        str: The file path for the architecture parameters.
    """
    nn_param_file_path: str = os.path.join(folder_path, "architecture.yaml")
    return nn_param_file_path


def create_nn(nn_type: NNModelType) -> ChiNN:
    """
    Create a neural network.
    """

    net: ChiNN
    match nn_type:
        case NNModelType.NetP1:
            net = NetP1()
        case NNModelType.NetP2:
            net = NetP2()
        case NNModelType.NetPP1:
            net = NetPP1()
        case NNModelType.NetPP2:
            net = NetPP2()
        case NNModelType.NetPP2D2:
            net = NetPP2D2()
        case NNModelType.NetPP2D2_2:
            net = NetPP2D2_2()
        case NNModelType.NetPP2D2_2_LEAKY:
            net = NetPP2D2_2_LEAKY()
        case NNModelType.NetPP2D2_2_RRELU:
            net = NetPP2D2_2_RRELU()
        case NNModelType.NetPP2D2_2_PRELU:
            net = NetPP2D2_2_PRELU()
        case NNModelType.TransformerOne:
            net = TransformerOne(n_embd=1, n_head=1, n_layer=1, dropout_ratio=0.0)
        case other:
            sys.exit(f"Create NN: can not find {other} in file {__name__}")
    return net


def get_architecture_args_from_file(
    architecture_file_name: path,
) -> NeuralNetArchitectureArgs:
    args_dict: dict[Any, Any] = yaml_fetch_args_in_file(
        path_file=architecture_file_name
    )
    print("debug args_dict", args_dict)
    nn_architecture_args: NeuralNetArchitectureArgs = dacite.from_dict(
        data_class=NeuralNetArchitectureArgs,
        data=args_dict,
        config=dacite.Config(cast=[Enum]),
    )
    return nn_architecture_args


def get_architecture_args_from_folder(folder_path: path) -> NeuralNetArchitectureArgs:
    architecture_file_name: path = get_nn_architecture_file_path_from(
        folder_path=folder_path
    )
    if not os.path.isfile(architecture_file_name):
        raise Exception(f"this is not a file {architecture_file_name}")

    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_file(
        architecture_file_name=architecture_file_name
    )

    return nn_architecture_args


def create_nn_from_folder_path_and_existing_model(
    folder_path: path,
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_folder(
        folder_path=folder_path
    )
    net: ChiNN = create_nn(nn_type=nn_architecture_args.model_type)
    model_weights_file_name: path = os.path.join(folder_path, "param.pt")
    net.load_weights_from_file(path_to_param_file=model_weights_file_name)
    return net, nn_architecture_args


def create_nn_board_eval_from_folder_path_and_existing_model(
    path_to_nn_folder: path,
) -> tuple[NNBoardEvaluator, NeuralNetArchitectureArgs]:
    """
    Create a neural network board evaluator.

    Args:
        path_to_nn_folder (path): the path to the folder where the model is defined.

    Returns:
        NNBoardEvaluator: The created neural network board evaluator.
    """
    net: ChiNN
    nn_architecture_args: NeuralNetArchitectureArgs
    net, nn_architecture_args = create_nn_from_folder_path_and_existing_model(
        folder_path=path_to_nn_folder
    )

    board_to_input_convert: BoardToInputFunction = create_board_to_input(
        model_input_representation_type=nn_architecture_args.model_input_representation_type
    )

    output_and_value_converter: OutputValueConverter = create_output_converter(
        model_output_type=nn_architecture_args.model_output_type
    )

    nn_board_evaluator = NNBoardEvaluator(
        net=net,
        output_and_value_converter=output_and_value_converter,
        board_to_input_convert=board_to_input_convert,
    )
    return nn_board_evaluator, nn_architecture_args


def create_nn_board_eval_from_architecture_args(
    nn_architecture_args: NeuralNetArchitectureArgs,
) -> NNBoardEvaluator:
    nn = create_nn(nn_type=nn_architecture_args.model_type)
    nn.init_weights()

    board_to_input_convert: BoardToInputFunction = create_board_to_input(
        model_input_representation_type=nn_architecture_args.model_input_representation_type
    )

    output_and_value_converter: OutputValueConverter = create_output_converter(
        model_output_type=nn_architecture_args.model_output_type
    )

    return NNBoardEvaluator(
        net=nn,
        output_and_value_converter=output_and_value_converter,
        board_to_input_convert=board_to_input_convert,
    )
