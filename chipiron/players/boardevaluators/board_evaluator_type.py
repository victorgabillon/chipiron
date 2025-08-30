"""
This module defines the different types of board evaluators used in the tree value calculation.
"""

from enum import Enum


class BoardEvalTypes(str, Enum):
    """
    Enumeration of argument types for different board evaluators.

    Attributes:
        NEURAL_NET_BOARD_EVAL: Argument type for neural network-based board evaluation.
        STOCKFISH_BOARD_EVAL: Argument type for Stockfish engine-based board evaluation.
        TABLE_BASE: Argument type for tablebase-based board evaluation.
        BASIC_EVALUATION: Argument type for basic/static board evaluation.
    """

    NEURAL_NET_BOARD_EVAL = "neural_network"
    STOCKFISH_BOARD_EVAL = "stockfish"
    TABLE_BASE_EVAL = "table_base"
    BASIC_EVALUATION_EVAL = "basic_evaluation"
