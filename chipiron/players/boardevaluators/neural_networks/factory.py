"""
Module for creating neural networks and neural network board evaluators.
"""

import os.path
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

import dacite
from sympy import N

from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    BoardToInputFunction,
    create_board_to_input,
)
from chipiron.players.boardevaluators.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
    MultiLayerPerceptronArgs,
)
from chipiron.players.boardevaluators.neural_networks.models.transformer_one import (
    TransformerArgs,
    TransformerOne,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.NNModelTypeArgs import (
    NNModelTypeArgs,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.factory import (
    create_output_converter,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import (
    OutputValueConverter,
)
from chipiron.utils import path, yaml_fetch_args_in_file
from chipiron.utils.chi_nn import ChiNN


@dataclass
class NeuralNetModelsAndArchitecture:
    """
    Class to hold the neural network models and architecture.
    Attributes:
        model_weights_file_name (path): The file name of the model weights.
        nn_architecture_args (NeuralNetArchitectureArgs): The neural network architecture arguments.
    """

    model_weights_file_name: path
    nn_architecture_args: NeuralNetArchitectureArgs

    @classmethod
    def build_from_folder_path(
        cls, folder_path: path
    ) -> "NeuralNetModelsAndArchitecture":
        """
        Build an instance of NeuralNetModelsAndArchitecture from the given folder path.

        Args:
            folder_path (Path): Path to the folder containing 'architecture.yaml' and model weights.

        Returns:
            NeuralNetModelsAndArchitecture: An initialized instance.
        """
        nn_args = get_architecture_args_from_folder(folder_path=folder_path)
        model_file = os.path.join(folder_path, nn_args.filename() + ".pt")

        return cls(model_weights_file_name=model_file, nn_architecture_args=nn_args)


def get_nn_param_file_path_from(
    folder_path: path, file_name: str | None = None
) -> tuple[str, str]:
    """
    Get the file path for the neural network parameters.

    Args:
        folder_path (str): The folder path for the neural network parameters.

    Returns:
        str: The file path for the neural network parameters.
    """
    nn_param_file_path: str
    if file_name is None:
        nn_param_file_path = os.path.join(folder_path, "param")
    else:
        nn_param_file_path = os.path.join(folder_path, file_name)
    return nn_param_file_path + ".pt", nn_param_file_path + ".yaml"


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


def create_nn(nn_type_args: NNModelTypeArgs) -> ChiNN:
    """
    Create a neural network.
    """

    net: ChiNN
    match nn_type_args:
        case MultiLayerPerceptronArgs():
            net = MultiLayerPerceptron(args=nn_type_args)
        case TransformerArgs():
            net = TransformerOne(args=nn_type_args)
        case other:
            sys.exit(f"Create NN: can not find {other} in file {__name__}")
    return net


# todo: probably dead code, check!
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


def create_nn_from_param_path_and_architecture_args(
    model_weights_file_name: path, nn_architecture_args: NeuralNetArchitectureArgs
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    net: ChiNN = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    net.load_weights_from_file(path_to_param_file=model_weights_file_name)
    return net, nn_architecture_args


def create_nn_from_folder_path_and_existing_model(
    folder_path: path,
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_folder(
        folder_path=folder_path
    )
    model_weights_file_name: path = os.path.join(folder_path, "param.pt")

    return create_nn_from_param_path_and_architecture_args(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
    )


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

    nn_board_evaluator = create_nn_board_eval_from_nn_and_architecture_args(
        nn=net, nn_architecture_args=nn_architecture_args
    )
    return nn_board_evaluator, nn_architecture_args


def create_nn_board_eval_from_nn_and_architecture_args(
    nn_architecture_args: NeuralNetArchitectureArgs,
    nn: ChiNN,
) -> NNBoardEvaluator:
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


def create_nn_board_eval_from_architecture_args(
    nn_architecture_args: NeuralNetArchitectureArgs,
) -> NNBoardEvaluator:
    nn = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    nn.init_weights()

    return create_nn_board_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args, nn=nn
    )


def create_nn_board_eval_from_nn_parameters_file_and_existing_model(
    model_weights_file_name: path, nn_architecture_args: NeuralNetArchitectureArgs
) -> NNBoardEvaluator:
    net: ChiNN
    net, nn_architecture_args = create_nn_from_param_path_and_architecture_args(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
    )

    return create_nn_board_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args, nn=net
    )
