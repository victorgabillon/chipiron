from dataclasses import dataclass
from enum import Enum

import chess


class FinalGameResult(Enum):
    WIN_FOR_WHITE = 0
    WIN_FOR_BLACK = 1
    DRAW = 2


@dataclass
class GameReport:
    final_game_result: FinalGameResult
    move_history: list[chess.Move]
