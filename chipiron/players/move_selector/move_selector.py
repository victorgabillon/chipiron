"""
This module defines the MoveSelector class and related data structures for selecting moves in a chess game.
"""

from dataclasses import dataclass
from typing import Protocol

import chess

import chipiron.environments.chess.board as boards
from chipiron.utils import seed
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation


@dataclass
class MoveRecommendation:
    """
    Represents a recommended move to play along with an optional evaluation score.
    """
    move: chess.Move
    evaluation: BoardEvaluation | None = None


class MoveSelector(Protocol):
    """
    Protocol for move selectors in a chess game.

    Move selectors are responsible for selecting the best move to play given a chess board and a move seed.
    """

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed: seed
    ) -> MoveRecommendation:
        """
        Selects the best move to play given a chess board and a move seed.

        Args:
            board: The current chess board.
            move_seed: The seed for move selection.

        Returns:
            The recommended move to play along with an optional evaluation score.
        """
        ...
