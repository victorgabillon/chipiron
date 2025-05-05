"""
This module provides a factory for creating neural network node evaluators.
"""

from dataclasses import dataclass, field
from typing import Literal

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    PointOfView,
)
from chipiron.players.boardevaluators.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.NNModelType import (
    ActivationFunctionType,
    NNModelType,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)
from chipiron.players.move_selector.treevalue.node_evaluator.all_node_evaluators import (
    NodeEvaluatorTypes,
)

from ..node_evaluator_args import NodeEvaluatorArgs


@dataclass
class NeuralNetNodeEvalArgs(NodeEvaluatorArgs):
    """
    Arguments for evaluating a node using a neural network.

    Attributes:
        path_to_nn_folder (path): Path to the folder containing the model weights and model architecture
    """

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
    type: Literal[NodeEvaluatorTypes.NeuralNetwork] = NodeEvaluatorTypes.NeuralNetwork

    def __post_init__(self) -> None:
        """
        Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.
        """
        if self.type != NodeEvaluatorTypes.NeuralNetwork:
            raise ValueError("Expecting neural_network as name")
        if (
            self.neural_nets_model_and_architecture.model_weights_file_name
            == "*default*"
        ):
            raise ValueError(f"Expecting a path_to_nn_folder in {__name__}")
