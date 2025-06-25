"""
Module that contains the NeuralNetBoardEvalArgs class.
"""

from dataclasses import dataclass

from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.players.boardevaluators.neural_networks.NNModelTypeArgs import (
    NNModelTypeArgs,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)


@dataclass
class NeuralNetArchitectureArgs:

    model_type_args: NNModelTypeArgs
    model_input_representation_type: ModelInputRepresentationType
    model_output_type: ModelOutputType

    def filename(self) -> str:
        # Get the filename part from the model type args (this uses the filename method from MultiLayerPerceptronArgs)
        model_type_filename = self.model_type_args.filename()

        # Get the string representation of the input and output types (Enum values)
        input_type = self.model_input_representation_type.value
        output_type = self.model_output_type.point_of_view.value

        # Return a formatted string that can be safely used as a filename but without defined extension
        return f"param_{model_type_filename}_{input_type}_{output_type}"


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
