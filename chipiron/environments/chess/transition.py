from dataclasses import dataclass

import chess
from chipiron.chessenvironment.board.board import BoardChi
from chipiron.chessenvironment.board.board_modification import BoardModification


@dataclass
class BoardTransition:
    board: BoardChi
    move: chess.Move
    next_board: BoardChi
    board_modifications: BoardModification
