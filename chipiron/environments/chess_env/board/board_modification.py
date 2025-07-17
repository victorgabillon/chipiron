"""
Module that contains the BoardModification class
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import chess


@dataclass(frozen=True)
class PieceInSquare:
    """
    Represents a piece on a chessboard square.
    """

    square: chess.Square
    piece: chess.PieceType
    color: chess.Color


class BoardModificationP(Protocol):
    """
    Represents a modification to a chessboard resulting from a move.
    """

    @property
    def removals(self) -> Iterator[PieceInSquare]: ...

    @property
    def appearances(self) -> Iterator[PieceInSquare]: ...


@dataclass
class BoardModification:
    """
    Represents a modification to a chessboard resulting from a move.
    """

    removals_: set[PieceInSquare] = field(default_factory=set)
    appearances_: set[PieceInSquare] = field(default_factory=set)

    def add_appearance(self, appearance: PieceInSquare) -> None:
        """
        Adds a piece appearance to the board modification.

        Args:
            appearance: The PieceInSquare object representing the appearance to add.
        """
        self.appearances_.add(appearance)

    def add_removal(self, removal: PieceInSquare) -> None:
        """
        Adds a piece removal to the board modification.

        Args:
            removal: The PieceInSquare object representing the removal to add.
        """
        self.removals_.add(removal)

    @property
    def removals(self) -> Iterator[PieceInSquare]:
        return iter(self.removals_)

    @property
    def appearances(self) -> Iterator[PieceInSquare]:
        return iter(self.appearances_)


@dataclass
class PieceRustIterator:
    items_: set[tuple[int, int, int]] = field(default_factory=set)

    def __iter__(self) -> Iterator[PieceInSquare]:
        self.it = iter(self.items_)
        return self

    def __next__(self) -> PieceInSquare:
        next_ = self.it.__next__()
        return PieceInSquare(square=next_[0], piece=next_[1], color=bool(next_[2]))


@dataclass
class BoardModificationRust:
    """
    Represents a modification to a chessboard resulting from a move.
    """

    removals_: set[tuple[int, int, int]] = field(default_factory=set)
    appearances_: set[tuple[int, int, int]] = field(default_factory=set)

    @property
    def removals(self) -> Iterator[PieceInSquare]:
        return PieceRustIterator(self.removals_)

    @property
    def appearances(self) -> Iterator[PieceInSquare]:
        return PieceRustIterator(self.appearances_)


def compute_modifications(
    previous_pawns: chess.Bitboard,
    previous_kings: chess.Bitboard,
    previous_queens: chess.Bitboard,
    previous_rooks: chess.Bitboard,
    previous_bishops: chess.Bitboard,
    previous_knights: chess.Bitboard,
    previous_occupied_white: chess.Bitboard,
    previous_occupied_black: chess.Bitboard,
    new_pawns: chess.Bitboard,
    new_kings: chess.Bitboard,
    new_queens: chess.Bitboard,
    new_rooks: chess.Bitboard,
    new_bishops: chess.Bitboard,
    new_knights: chess.Bitboard,
    new_occupied_white: chess.Bitboard,
    new_occupied_black: chess.Bitboard,
) -> BoardModification:
    board_modifications: BoardModification = BoardModification()
    hop = [
        (previous_pawns, new_pawns, chess.PAWN),
        (previous_bishops, new_bishops, chess.BISHOP),
        (previous_rooks, new_rooks, chess.ROOK),
        (previous_knights, new_knights, chess.KNIGHT),
        (previous_queens, new_queens, chess.QUEEN),
        (previous_kings, new_kings, chess.KING),
    ]
    hip = [
        (previous_occupied_white, new_occupied_white, chess.WHITE),
        (previous_occupied_black, new_occupied_black, chess.BLACK),
    ]

    for previous_bitboard_piece, new_bitboard_piece, piece_type in hop:
        for previous_bitboard_color, new_bitboard_color, color in hip:

            removals: chess.Bitboard = (
                previous_bitboard_piece & previous_bitboard_color
            ) & ~(new_bitboard_piece & new_bitboard_color)
            if removals:
                for square in chess.scan_forward(removals):
                    board_modifications.add_removal(
                        PieceInSquare(square=square, piece=piece_type, color=color)
                    )

            appearance: chess.Bitboard = ~(
                previous_bitboard_piece & previous_bitboard_color
            ) & (new_bitboard_piece & new_bitboard_color)

            if appearance:
                for square in chess.scan_forward(appearance):
                    board_modifications.add_appearance(
                        PieceInSquare(square=square, piece=piece_type, color=color)
                    )

    return board_modifications
