"""
This module defines the arguments for different board evaluators in the Chipiron chess engine framework.
"""

from dataclasses import dataclass
from typing import Literal

from chipiron.players.boardevaluators.board_evaluator_type import BoardEvalTypes
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetBoardEvalArgs,
)
from chipiron.players.boardevaluators.stockfish_board_evaluator import (
    StockfishBoardEvalArgs,
)


@dataclass
class TableBaseArgs:
    """A class representing the arguments for the TableBase class.

    This class provides a template for the arguments that can be passed to the TableBase class.
    It serves as a base class for defining specific argument classes for different implementations
    of the TableBase class.

    Attributes:
        ``None``

    Methods:
        ``None``
    """

    type: Literal[BoardEvalTypes.TABLE_BASE_EVAL]


@dataclass
class BasicEvaluationBoardEvaluatorArgs:
    """
    Represents the arguments for a board evaluator.

    """

    type: Literal[BoardEvalTypes.BASIC_EVALUATION_EVAL]


AllBoardEvaluatorArgs = (
    NeuralNetBoardEvalArgs
    | StockfishBoardEvalArgs
    | TableBaseArgs
    | BasicEvaluationBoardEvaluatorArgs
)
