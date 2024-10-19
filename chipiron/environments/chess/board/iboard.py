from dataclasses import asdict
from typing import Protocol, Self

import chess
import yaml

from chipiron.environments.chess.board.board_modification import BoardModification
from .utils import FenPlusMoveHistory
from .utils import fen

board_key = tuple[int, int, int, int, int, int, bool, int, int | None, int, int, int, int, int]


class IBoard(Protocol):
    fast_representation_: board_key | None = None

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

    @property
    def pawns(self) -> chess.Bitboard:
        ...

    @property
    def knights(self) -> chess.Bitboard:
        ...

    @property
    def bishops(self) -> chess.Bitboard:
        ...

    @property
    def rooks(self) -> chess.Bitboard:
        ...

    @property
    def queens(self) -> chess.Bitboard:
        ...

    @property
    def kings(self) -> chess.Bitboard:
        ...

    @property
    def white(self) -> chess.Bitboard:
        ...

    @property
    def black(self) -> chess.Bitboard:
        ...

    @property
    def halfmove_clock(self) -> int:
        ...

    @property
    def promoted(self) -> chess.Bitboard:
        ...

    @property
    def fullmove_number(self) -> int:
        ...

    @property
    def castling_rights(self) -> chess.Bitboard:
        ...

    @property
    def occupied(self) -> chess.Bitboard:
        ...

    def occupied_color(self, color: chess.COLORS) -> chess.Bitboard:
        ...

    def result(self) -> str:
        ...

    def termination(self) -> chess.Termination | None:
        ...

    def dump(self, file) -> None:
        # create minimal info for reconstruction that is the class FenPlusMoveHistory

        current_fen: fen = self.fen
        fen_plus_moves: FenPlusMoveHistory = FenPlusMoveHistory(
            current_fen=current_fen,
            historical_moves=self.move_history_stack
        )

        yaml.dump(asdict(fen_plus_moves), file, default_flow_style=False)

    @property
    def ep_square(self) -> int | None:
        ...

    def compute_key(self) -> board_key:
        """
        Computes and returns a unique key representing the current state of the chess board.

        The key is computed by concatenating various attributes of the board, including the positions of pawns, knights,
        bishops, rooks, queens, and kings, as well as the current turn, castling rights, en passant square, halfmove clock,
        occupied squares for each color, promoted pieces, and the fullmove number.
        It is faster than calling the fen.
        Returns:
            str: A unique key representing the current state of the chess board.
        """
        string = (self.pawns, self.knights, self.bishops, self.rooks, self.queens, self.kings,
                  self.turn, self.castling_rights, self.ep_square,
                  self.white, self.black, self.promoted, self.fullmove_number, self.halfmove_clock)
        return string


    @property
    def fast_representation(self) -> str:
        """
        Returns a fast representation of the board.

        This method computes and returns a string representation of the board
        that can be quickly generated and used for various purposes.

        :return: A string representation of the board.
        :rtype: str
        """
        return self.fast_representation_


    @property
    def fast_representation_without_counters(self) -> str:
        """
        Returns a fast representation of the board.

        This method computes and returns a string representation of the board
        that can be quickly generated and used for various purposes.

        :return: A string representation of the board.
        :rtype: str
        """
        return self.fast_representation_[:-2]