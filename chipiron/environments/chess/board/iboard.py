from typing import Protocol, Self

import chess
import yaml

from chipiron.environments.chess.board.board_modification import BoardModification
from .utils import FenPlusMoveHistory
from .utils import fen
from dataclasses import asdict

class IBoard(Protocol):

    def play_move(
            self,
            move: chess.Move
    ) -> BoardModification | None:
        ...

    @property
    def fen(self) -> str:
        ...

    @property
    def move_history_stack(
            self,
    ) -> list[chess.Move]:
        ...

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        ...

    @property
    def turn(self) -> chess.Color:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        ...

    def copy(
            self,
            stack: bool
    ) -> Self:
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            BoardChi: A new instance of the BoardChi class with the copied board.
        """
        ...

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        ...

    def dump(self, file) -> None:
        # create minimal info for reconstruction that is the class FenPlusMoveHistory

        current_fen: fen = self.fen
        fen_plus_moves: FenPlusMoveHistory = FenPlusMoveHistory(
            current_fen=current_fen,
            historical_moves=self.move_history_stack
        )

        yaml.dump(asdict(fen_plus_moves), file, default_flow_style=False)
