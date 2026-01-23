"""
This module defines the different types of board evaluators used in the tree value calculation.
"""

from enum import Enum

from coral.neural_networks.neural_net_board_eval_args import NN_NET_EVAL_STRING


class BoardEvalTypes(str, Enum):
    """
    Enumeration of argument types for different board evaluators.

    Attributes:
        NEURAL_NET_BOARD_EVAL: Argument type for neural network-based board evaluation.
        STOCKFISH_BOARD_EVAL: Argument type for Stockfish engine-based board evaluation.
        TABLE_BASE: Argument type for tablebase-based board evaluation.
        BASIC_EVALUATION: Argument type for basic/static board evaluation.
    """

    NEURAL_NET_BOARD_EVAL = NN_NET_EVAL_STRING
    STOCKFISH_BOARD_EVAL = "stockfish"
    TABLE_BASE_EVAL = "table_base"
    BASIC_EVALUATION_EVAL = "basic_evaluation"


def to_board_eval_type(value: str) -> BoardEvalTypes:
    return BoardEvalTypes(value)
