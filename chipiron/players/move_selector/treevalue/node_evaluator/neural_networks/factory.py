"""
This module provides a factory for creating neural network node evaluators.
"""

from dataclasses import dataclass

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
        path_to_folder_directory (path): Path to the folder containing the model weights and model architecture
    """

    path_to_nn_folder: path

    def __post_init__(self) -> None:
        """
        Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.
        """
        if self.type != NodeEvaluatorTypes.NeuralNetwork:
            raise ValueError("Expecting neural_network as name")
