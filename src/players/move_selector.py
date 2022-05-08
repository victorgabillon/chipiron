from typing import Protocol
from src.chessenvironment.board.iboard import IBoard


class MoveSelector(Protocol):

    def select_move(self, board: IBoard):
        ...
