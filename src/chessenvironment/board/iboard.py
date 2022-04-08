from typing import Protocol
import chess


class IBoard(Protocol):
    """
    This class is the interface for a board
    """

    def ply(self):
        ...

    @property
    def turn(self) -> int:
        ...

    @property
    def legal_moves(self):
        ...

    def play_move(self, move: chess.Move) -> None:
        ...

    def rewind_one_move(self) -> None:
        ...

    def is_game_over(self):
        ...

    def fast_representation(self) -> str:
        ...
