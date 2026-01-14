from dataclasses import dataclass

from coral.neural_networks.nn_model_type_args import (
    NNModelTypeArgs,
)
from coral.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)

from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)


@dataclass
class NeuralNetArchitectureArgs:
    """
    Represents the arguments for a neural network architecture.

    Attributes:
        model_type_args (NNModelTypeArgs): Arguments specific to the model type.
        model_input_representation_type (ModelInputRepresentationType): Type of input representation for the model.
        model_output_type (ModelOutputType): Type of output produced by the model.
    """

    model_type_args: NNModelTypeArgs
    model_input_representation_type: ModelInputRepresentationType
    model_output_type: ModelOutputType

    def filename(self) -> str:
        """
        Generates a filename string based on the model type arguments, input representation type, and output type.
        Returns:
            str: A formatted string suitable for use as a filename, composed of the model type filename,
                 input representation type, and output type, without a file extension.
        """

        # Get the filename part from the model type args (this uses the filename method from MultiLayerPerceptronArgs)
        model_type_filename = self.model_type_args.filename()

        # Get the string representation of the input and output types (Enum values)
        input_type = self.model_input_representation_type.value
        output_type = self.model_output_type.point_of_view.value

        # Return a formatted string that can be safely used as a filename but without defined extension
        return f"param_{model_type_filename}_{input_type}_{output_type}"
