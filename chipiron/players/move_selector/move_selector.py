"""
This module defines the MoveSelector class and related data structures for selecting moves in a chess game.
"""

from dataclasses import dataclass
from typing import Protocol

from chipiron.environments.chess.board import IBoard
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    BoardEvaluation,
)
from chipiron.utils import seed


@dataclass
class MoveRecommendation:
    """
    Represents a recommended move to play along with an optional evaluation score.
    """

    # todo should it be a movekey or moveuci? both?
    move: moveKey
    evaluation: BoardEvaluation | None = None


class MoveSelector(Protocol):
    """
    Protocol for move selectors in a chess game.

    Move selectors are responsible for selecting the best move to play given a chess board and a move seed.
    """

    def select_move(self, board: IBoard, move_seed: seed) -> MoveRecommendation:
        """
        Selects the best move to play given a chess board and a move seed.

        Args:
            board: The current chess board.
            move_seed: The seed for move selection.

        Returns:
            The recommended move to play along with an optional evaluation score.
        """
        ...
