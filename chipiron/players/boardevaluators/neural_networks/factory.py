"""
Module for creating neural networks and neural network board evaluators.
"""

import os.path
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import dacite
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
    TransformerOne,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_content_evaluator import (
    NNBWContentEvaluator,
)
from coral.neural_networks.nn_model_type_args import (
    NNModelTypeArgs,
)
from coral.neural_networks.output_converters.factory import (
    create_output_converter,
)

from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    create_board_to_input,
)
from chipiron.utils import path, yaml_fetch_args_in_file
from coral.chi_nn import ChiNN

if TYPE_CHECKING:
    from coral.neural_networks.output_converters.output_value_converter import (
        OutputValueConverter,
    )


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
        case _:
            sys.exit(f"Create NN: can not find {nn_type_args} in file {__name__}")
    return net


# todo: probably dead code, check!
def get_architecture_args_from_file(
    architecture_file_name: path,
) -> NeuralNetArchitectureArgs:
    """Get the architecture arguments from a YAML file.

    Args:
        architecture_file_name (path): The path to the architecture YAML file.

    Returns:
        NeuralNetArchitectureArgs: The architecture arguments.
    """
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
    """Get the architecture arguments from a folder.
    Args:
        folder_path (path): The path to the folder containing the architecture file.
    Returns:
        NeuralNetArchitectureArgs: The architecture arguments.
    """
    architecture_file_name: path = get_nn_architecture_file_path_from(
        folder_path=folder_path
    )
    if not os.path.isfile(architecture_file_name):
        raise FileNotFoundError(f"this is not a file {architecture_file_name}")

    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_file(
        architecture_file_name=architecture_file_name
    )

    return nn_architecture_args


def create_nn_from_param_path_and_architecture_args(
    model_weights_file_name: path, nn_architecture_args: NeuralNetArchitectureArgs
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    """Create a neural network from a parameter file and architecture arguments.
    Args:
        model_weights_file_name (path): The path to the model weights file.
        nn_architecture_args (NeuralNetArchitectureArgs): The architecture arguments.
    Returns:
        tuple[ChiNN, NeuralNetArchitectureArgs]: The created neural network and the architecture arguments.
    """
    net: ChiNN = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    net.load_weights_from_file(path_to_param_file=model_weights_file_name)
    return net, nn_architecture_args


def create_nn_from_folder_path_and_existing_model(
    folder_path: path,
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    """Create a neural network from a folder path and existing model.

    Args:
        folder_path (path): The path to the folder containing the model files.

    Returns:
        tuple[ChiNN, NeuralNetArchitectureArgs]: _description_
    """
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
) -> tuple[NNBWContentEvaluator, NeuralNetArchitectureArgs]:
    """
    Create a neural network board evaluator.

    Args:
        path_to_nn_folder (path): the path to the folder where the model is defined.

    Returns:
        NNBWContentEvaluator: The created neural network board evaluator.
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
) -> NNBWContentEvaluator:
    board_to_input_convert: ContentToInputFunction = create_board_to_input(
        model_input_representation_type=nn_architecture_args.model_input_representation_type
    )

    output_and_value_converter: OutputValueConverter = create_output_converter(
        model_output_type=nn_architecture_args.model_output_type
    )

    return NNBWContentEvaluator(
        net=nn,
        output_and_value_converter=output_and_value_converter,
        board_to_input_convert=board_to_input_convert,
    )


def create_nn_board_eval_from_architecture_args(
    nn_architecture_args: NeuralNetArchitectureArgs,
) -> NNBWContentEvaluator:
    nn = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    nn.init_weights()

    return create_nn_board_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args, nn=nn
    )


def create_nn_board_eval_from_nn_parameters_file_and_existing_model(
    model_weights_file_name: path, nn_architecture_args: NeuralNetArchitectureArgs
) -> NNBWContentEvaluator:
    net: ChiNN
    net, nn_architecture_args = create_nn_from_param_path_and_architecture_args(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
    )

    return create_nn_board_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args, nn=net
    )
