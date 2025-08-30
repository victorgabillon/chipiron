"""
Module that contains the NeuralNetBoardEvalArgs class.
"""

from dataclasses import dataclass, field
from typing import Literal

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    PointOfView,
)
from chipiron.players.boardevaluators.board_evaluator_type import BoardEvalTypes
from chipiron.players.boardevaluators.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import (
    ActivationFunctionType,
    NNModelType,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)


@dataclass
class NeuralNetBoardEvalArgs:
    """
    NeuralNetBoardEvalArgs encapsulates the configuration arguments required for evaluating board positions
    using a neural network-based node evaluator.

    Attributes:
        neural_nets_model_and_architecture (NeuralNetModelsAndArchitecture):
            Specifies the neural network model, its architecture, and associated parameters.
            Defaults to a MultiLayerPerceptron with predefined layer sizes and activation functions.
        type (Literal[NodeEvaluatorTypes.NeuralNetwork]):
            The type of node evaluator, which must be set to 'NeuralNetwork'.
    """

    type: Literal[BoardEvalTypes.NEURAL_NET_BOARD_EVAL]
    neural_nets_model_and_architecture: NeuralNetModelsAndArchitecture = field(
        default_factory=lambda: NeuralNetModelsAndArchitecture(
            model_weights_file_name="*default*",
            nn_architecture_args=NeuralNetArchitectureArgs(
                model_type_args=MultiLayerPerceptronArgs(
                    type=NNModelType.MultiLayerPerceptron,
                    number_neurons_per_layer=[5, 1],
                    list_of_activation_functions=[
                        ActivationFunctionType.TangentHyperbolic
                    ],
                ),
                model_output_type=ModelOutputType(
                    point_of_view=PointOfView.PLAYER_TO_MOVE
                ),
                model_input_representation_type=ModelInputRepresentationType.PIECE_DIFFERENCE,
            ),
        )
    )

    def __post_init__(self) -> None:
        """
        Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.
        """
        if self.type != BoardEvalTypes.NEURAL_NET_BOARD_EVAL:
            raise ValueError("Expecting neural_network as name")
        if (
            self.neural_nets_model_and_architecture.model_weights_file_name
            == "*default*"
        ):
            raise ValueError(f"Expecting a path_to_nn_folder in {__name__}")
