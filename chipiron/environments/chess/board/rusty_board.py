from dataclasses import dataclass
from typing import Self

import chess
import shakmaty_python_binding

from chipiron.environments.chess.board.board_modification import BoardModification


@dataclass
class RustyBoardChi:
    """
    Rusty Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness

    This is the Rust version for speedy execution
    """

    chess_: shakmaty_python_binding.MyChess

    def play_move(
            self,
            move: chess.Move
    ) -> BoardModification | None:
        self.chess_.play(move.uci())
        return None

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        return self.chess_.ply()

    @property
    def turn(self) -> chess.Color:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        return bool(self.chess_.turn())

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        #todo add move stack chek for repetition
        return self.chess_.is_game_over()

    def copy(
            self,
            stack: bool
    ) -> Self:
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            RustyBoardChi: A new instance of the BoardChi class with the copied board.
        """
        #todo move stack !!
        chess_copy: shakmaty_python_binding.MyChess = self.chess_.copy()
        return RustyBoardChi(
            chess_=chess_copy
        )
