from dataclasses import dataclass
from chipiron.chessenvironment.board.board import BoardChi
from chipiron.chessenvironment.board.board_modification import BoardModification
import chess


@dataclass
class BoardTransition:
    board: BoardChi
    move: chess.Move
    next_board: BoardChi
    board_modifications: BoardModification
