from dataclasses import dataclass, field

import chess


@dataclass(frozen=True)
class PieceInSquare:
    square: chess.Square
    piece: chess.PieceType
    color: chess.Color


@dataclass
class BoardModification:
    """
    object that describes the modification to a board from a move
    """

    removals: set[PieceInSquare] = field(default_factory=set)
    appearances: set[PieceInSquare] = field(default_factory=set)

    def add_appearance(
            self,
            appearance: PieceInSquare
    ) -> None:
        self.appearances.add(appearance)

    def add_removal(
            self,
            removal: PieceInSquare
    ) -> None:
        self.removals.add(removal)
