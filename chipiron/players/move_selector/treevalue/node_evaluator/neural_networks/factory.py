"""
This module provides a factory for creating neural network node evaluators.
"""

from dataclasses import dataclass

import chipiron.players.boardevaluators.neural_networks as board_nn
from chipiron.players.move_selector.treevalue.node_evaluator.all_node_evaluators import (
    NodeEvaluatorTypes,
)

from ..node_evaluator_args import NodeEvaluatorArgs


@dataclass
class NeuralNetNodeEvalArgs(NodeEvaluatorArgs):
    """
    Arguments for evaluating a node using a neural network.

    Attributes:
        neural_network (board_nn.NeuralNetBoardEvalArgs): The neural network used for evaluation.
    """

    neural_network: board_nn.NeuralNetBoardEvalArgs

    def __post_init__(self) -> None:
        """
        Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.
        """
        if self.type != NodeEvaluatorTypes.NeuralNetwork:
            raise ValueError('Expecting neural_network as name')
