from typing import Protocol
import chipiron.environments.chess.board as boards


class MoveSelector(Protocol):

    def select_move(
            self,
            board: boards.BoardChi
    ):
        ...
