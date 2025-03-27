"""
This module provides a factory for creating neural network node evaluators.
"""

from dataclasses import dataclass

from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.move_selector.treevalue.node_evaluator.all_node_evaluators import (
    NodeEvaluatorTypes,
)
from chipiron.utils import path
from ..node_evaluator_args import NodeEvaluatorArgs


@dataclass
class NeuralNetNodeEvalArgs(NodeEvaluatorArgs):
    """
    Arguments for evaluating a node using a neural network.

    Attributes:
        path_to_nn_folder (path): Path to the folder containing the model weights and model architecture
    """

    path_to_nn_folder: path = "*default*"
    model_input_representation_type: ModelInputRepresentationType = (
        ModelInputRepresentationType.PIECE_MAP
    )

    def __post_init__(self) -> None:
        """
        Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.
        """
        if self.type != NodeEvaluatorTypes.NeuralNetwork:
            raise ValueError("Expecting neural_network as name")
        if self.path_to_nn_folder == "*default*":
            raise ValueError(f"Expecting a path_to_nn_folder in {__name__}")
