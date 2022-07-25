from typing import Protocol
from chipiron.chessenvironment.board.iboard import IBoard


class MoveSelector(Protocol):

    def select_move(self, board: IBoard):
        ...
