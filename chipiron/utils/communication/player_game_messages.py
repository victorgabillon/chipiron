"""
Module to define the messages that are sent between the players and the game.
"""
from dataclasses import dataclass

import chess

from chipiron.environments.chess.board import fen
from chipiron.environments.chess.board.board import BoardChi


@dataclass
class MoveMessage:
    """
    Represents a message containing a move made by a player.

    Attributes:
        move (chess.Move): The move made by the player.
        corresponding_board (fen): The FEN representation of the board after the move.
        player_name (str): The name of the player who made the move.
        color_to_play (chess.Color): The color of the player to play after the move.
        evaluation (float | None, optional): The evaluation score of the move. Defaults to None.
    """
    move: chess.Move
    corresponding_board: fen
    player_name: str
    color_to_play: chess.Color
    evaluation: float | None = None


@dataclass
class BoardMessage:
    """
    Represents a message containing the current state of the chess board.

    Attributes:
        board (BoardChi): The current state of the chess board.
        seed (int | None, optional): The seed used for random number generation. Defaults to None.
    """
    board: BoardChi
    seed: int | None = None
