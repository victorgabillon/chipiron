"""
This module defines the MoveSelector class and related data structures for selecting moves in a chess game.
"""

from typing import Protocol

from atomheart.board import IBoard

from chipiron.utils import Seed


class MoveSelector(Protocol):
    """
    Protocol for move selectors in a chess game.

    Move selectors are responsible for selecting the best move to play given a chess board and a move seed.
    """

    def select_move(self, board: IBoard, move_seed: Seed) -> MoveRecommendation:
        """
        Selects the best move to play given a chess board and a move seed.

        Args:
            board: The current chess board.
            move_seed: The seed for move selection.

        Returns:
            The recommended move to play along with an optional evaluation score.
        """
        ...
