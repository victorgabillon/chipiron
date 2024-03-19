from dataclasses import dataclass

import chess

from chipiron.environments.chess.board import fen
from chipiron.environments.chess.board.board import BoardChi


@dataclass
class MoveMessage:
    move: chess.Move
    corresponding_board: fen
    player_name: str
    color_to_play: chess.Color
    evaluation: float | None = None


@dataclass
class BoardMessage:
    board: BoardChi
    seed: int | None = None
