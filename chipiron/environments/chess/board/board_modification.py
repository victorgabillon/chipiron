"""
Module that contains the BoardModification class
"""
from dataclasses import dataclass, field

import chess


@dataclass(frozen=True)
class PieceInSquare:
    """
    Represents a piece on a chessboard square.
    """

    square: chess.Square
    piece: chess.PieceType
    color: chess.Color


@dataclass
class BoardModification:
    """
    Represents a modification to a chessboard resulting from a move.
    """

    removals: set[PieceInSquare] = field(default_factory=set)
    appearances: set[PieceInSquare] = field(default_factory=set)

    def add_appearance(
            self,
            appearance: PieceInSquare
    ) -> None:
        """
        Adds a piece appearance to the board modification.

        Args:
            appearance: The PieceInSquare object representing the appearance to add.
        """
        self.appearances.add(appearance)

    def add_removal(
            self,
            removal: PieceInSquare
    ) -> None:
        """
        Adds a piece removal to the board modification.

        Args:
            removal: The PieceInSquare object representing the removal to add.
        """
        self.removals.add(removal)
