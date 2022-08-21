from typing import Protocol
import chipiron as ch


class MoveSelector(Protocol):

    def select_move(self, board: ch.chess.board.BoardChi):
        ...
