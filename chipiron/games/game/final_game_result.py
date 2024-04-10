""" 
Module for the FinalGameResult enum and the GameReport dataclass.
"""

from dataclasses import dataclass
from enum import Enum

import chess


class FinalGameResult(Enum):
    """Enum representing the final result of a game."""
    WIN_FOR_WHITE = 0
    WIN_FOR_BLACK = 1
    DRAW = 2


@dataclass
class GameReport:
    """Dataclass representing a game report."""
    final_game_result: FinalGameResult
    move_history: list[chess.Move]
