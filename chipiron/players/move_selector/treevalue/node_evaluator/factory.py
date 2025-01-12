"""
This module provides a factory function for creating node evaluators based on different types of board evaluators.
"""

import sys
from typing import Any, TypeAlias

import chipiron.players.boardevaluators.basic_evaluation as basic_evaluation
from chipiron.players.boardevaluators.neural_networks.factory import (
    create_nn_board_eval,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable

from . import neural_networks
from .all_node_evaluators import NodeEvaluatorTypes
from .neural_networks.nn_node_evaluator import NNNodeEvaluator
from .node_evaluator import NodeEvaluator
from .node_evaluator_args import NodeEvaluatorArgs

AllNodeEvaluatorArgs: TypeAlias = neural_networks.NeuralNetNodeEvalArgs | NodeEvaluatorArgs


def create_node_evaluator(
        arg_board_evaluator: AllNodeEvaluatorArgs,
        syzygy: SyzygyTable[Any] | None
) -> NodeEvaluator:
    """
    Create a node evaluator based on the given board evaluator argument and syzygy table.

    Args:
        arg_board_evaluator (AllNodeEvaluatorArgs): The argument for the board evaluator.
        syzygy (SyzygyTable | None): The syzygy table to be used for evaluation.

    Returns:
        NodeEvaluator: The created node evaluator.

    Raises:
        SystemExit: If the given board evaluator type is not found.

    """
    if arg_board_evaluator.syzygy_evaluation:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    node_evaluator: NodeEvaluator

    match arg_board_evaluator.type:
        case 'basic_evaluation':
            board_evaluator: basic_evaluation.BasicEvaluation = basic_evaluation.BasicEvaluation()
            node_evaluator = NodeEvaluator(
                board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case NodeEvaluatorTypes.NeuralNetwork:
            assert isinstance(arg_board_evaluator, neural_networks.NeuralNetNodeEvalArgs)
            board_evaluator_nn: NNBoardEvaluator = create_nn_board_eval(
                arg=arg_board_evaluator.neural_network,
                representation_type=arg_board_evaluator.representation_type
            )
            node_evaluator = NNNodeEvaluator(
                nn_board_evaluator=board_evaluator_nn,
                syzygy=syzygy_
            )
        case other:
            sys.exit(f'Node Board Eval: can not find {other} in file {__name__}')

    return node_evaluator
