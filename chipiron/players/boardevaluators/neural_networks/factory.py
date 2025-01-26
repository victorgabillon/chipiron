"""
Module for creating neural networks and neural network board evaluators.
"""

import os.path
import sys
from enum import Enum
from typing import Any

import dacite

from chipiron.players.boardevaluators.neural_networks import NeuralNetBoardEvalArgs
from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.input_converters.TensorRepresentationType import (
    InternalTensorRepresentationType,
    compatibilities,
    assert_compatibilities_representation_type,
)
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
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import (
    OneDToValueWhite,
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
        case "pp1":
            net = NetPP1()
        case "pp2":
            net = NetPP2()
        case "pp2d2":
            net = NetPP2D2()
        case "pp2d2_2":
            net = NetPP2D2_2()
        case "pp2d2_2_leaky":
            net = NetPP2D2_2_LEAKY()
        case "pp2d2_2_rrelu":
            net = NetPP2D2_2_RRELU()
        case "pp2d2_2_prelu":
            net = NetPP2D2_2_PRELU()
        case other:
            sys.exit(f"Create NN: can not find {other} in file {__name__}")
    return net


def get_architecture_args_from_file(
    architecture_file_name: path,
) -> NeuralNetArchitectureArgs:
    args_dict: dict[Any, Any] = yaml_fetch_args_in_file(
        path_file=architecture_file_name
    )
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


def create_nn_board_eval(
    path_to_nn_folder: path,
    internal_representation_type: InternalTensorRepresentationType,
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
    net: ChiNN
    net, nn_architecture_args = create_nn_from_folder_path_and_existing_model(
        folder_path=path_to_nn_folder
    )

    assert_compatibilities_representation_type(
        tensor_representation_type=nn_architecture_args.tensor_representation_type,
        internal_tensor_representation_type=internal_representation_type,
    )

    output_and_value_converter: OutputValueConverter = OneDToValueWhite(
        point_of_view=net.evaluation_point_of_view
    )
    representation_factory: RepresentationFactory[Any] | None = (
        create_board_representation_factory(
            board_representation_factory_type=internal_representation_type
        )
    )
    assert representation_factory is not None
    board_to_input_converter: BoardToInput = RepresentationBTI(
        representation_factory=representation_factory
    )
    return NNBoardEvaluator(
        net=net,
        output_and_value_converter=output_and_value_converter,
        board_to_input_converter=board_to_input_converter,
    )
