"""
This module provides a factory function for creating node evaluators based on different types of board evaluators.
"""

import sys
from typing import Any

from chipiron.players.boardevaluators import MasterBoardEvaluator
from chipiron.players.boardevaluators.board_evaluator_type import BoardEvalTypes
from chipiron.players.boardevaluators.master_board_evaluator import (
    create_master_board_evaluator_from_args,
)
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.players.move_selector.treevalue.node_evaluator.node_evaluator_args import (
    NodeEvaluatorArgs,
)

from .neural_networks.nn_node_evaluator import NNNodeEvaluator
from .node_evaluator import NodeEvaluator


def create_node_evaluator(
    arg_board_evaluator: NodeEvaluatorArgs, syzygy: SyzygyTable[Any] | None
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

    master_board_evaluator: MasterBoardEvaluator = (
        create_master_board_evaluator_from_args(
            arg_board_evaluator.master_board_evaluator,
            syzygy,
        )
    )

    match arg_board_evaluator.master_board_evaluator.board_evaluator.type:
        case BoardEvalTypes.BASIC_EVALUATION_EVAL:
            node_evaluator = NodeEvaluator(
                master_board_evaluator=master_board_evaluator
            )
        case BoardEvalTypes.NEURAL_NET_BOARD_EVAL:
            node_evaluator = NNNodeEvaluator(
                master_board_evaluator=master_board_evaluator
            )
        case other:
            sys.exit(f"Node Board Eval: can not find {other} in file {__name__}")

    return node_evaluator
