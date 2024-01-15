from dataclasses import dataclass
import chess
from chipiron.environments.chess.board import fen, BoardChi


@dataclass
class MoveMessage:
    move: chess.Move
    corresponding_board: fen
    player_name: str
    color_to_play : chess.COLORS
    evaluation: float = None


@dataclass
class BoardMessage:
    board: BoardChi
