"""
Module to define the transition between two boards
"""

from dataclasses import dataclass

import chess

from chipiron.environments.chess.board.board_chi import BoardChi
from chipiron.environments.chess.board.board_modification import BoardModification


@dataclass
class BoardTransition:
    """
    Represents a transition from one chess board state to another.

    Attributes:
        board (BoardChi): The initial chess board state.
        move (chess.Move): The move that was made to transition to the next board state.
        next_board (BoardChi): The resulting chess board state after the move.
        board_modifications (BoardModification): The modifications made to the board during the transition.
    """

    board: BoardChi
    move: chess.Move
    next_board: BoardChi
    board_modifications: BoardModification
