"""
Module for the FinalGameResult enum and the GameReport dataclass.
"""

from dataclasses import dataclass
from enum import Enum

from chipiron.environments.chess.board.utils import fen
from chipiron.environments.chess.move import moveUci


class FinalGameResult(str, Enum):
    """Enum representing the final result of a game."""

    WIN_FOR_WHITE = "win_for_white"
    WIN_FOR_BLACK = "win_for_black"
    DRAW = "draw"


@dataclass
class GameReport:
    """Dataclass representing a game report."""

    final_game_result: FinalGameResult
    move_history: list[moveUci]
    fen_history: list[fen]
