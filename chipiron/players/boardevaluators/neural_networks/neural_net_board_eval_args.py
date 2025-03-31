"""
Module that contains the NeuralNetBoardEvalArgs class.
"""

from dataclasses import dataclass

from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)


@dataclass
class NeuralNetArchitectureArgs:
    """ """

    model_type: NNModelType
    model_input_representation_type: ModelInputRepresentationType
    model_output_type: ModelOutputType


@dataclass
class NeuralNetBoardEvalArgs:
    """
    Represents the arguments for a neural network board evaluator.

    Attributes:
        nn_type (str): The type of the neural network.
        nn_param_folder_name (str): The name of the folder containing the neural network parameters.
    """

    nn_type: NNModelType
    nn_param_folder_name: str
