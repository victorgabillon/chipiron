"""Module to define the messages that are sent between the players and the game."""

from dataclasses import dataclass

from atomheart.games.chess.board.utils import Fen
from atomheart.games.chess.move import MoveUci
from valanga import Color, TurnState
from valanga.evaluations import (
    StateEvaluation,
)
from valanga.game import TurnStatePlusHistory

from chipiron.players.player import PlayerId


@dataclass
class MoveMessage:
    """Represents a message containing a move made by a player.

    Attributes:
        move (chess.Move): The move made by the player.
        corresponding_board (fen): The FEN representation of the board after the move.
        player_name (str): The name of the player who made the move.
        color_to_play (chess.Color): The color of the player to play after the move.
        evaluation (float | None, optional): The evaluation score of the move. Defaults to None.

    """

    move: MoveUci
    corresponding_board: Fen
    player_name: PlayerId
    color_to_play: Color
    evaluation: StateEvaluation | None = None


@dataclass
class StatePlusHistoryMessage[StateT: TurnState = TurnState]:
    """Represents a message containing the current state of the chess board.

    Attributes:
        fen_plus_moves (FenPlusMoves): The original fen and the subsequent moves to be applied.
        seed (int | None, optional): The seed used for random number generation. Defaults to None.

    """

    state_plus_history: TurnStatePlusHistory[StateT]
    seed: int | None = None
