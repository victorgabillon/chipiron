"""Utility functions and data structures for handling chess board states and FEN strings."""

import typing
from dataclasses import dataclass, field

import chess

from chipiron.environments.chess_env.move import moveUci

fen = typing.Annotated[str, "a string representing a fen"]


@dataclass
class FenPlusMoves:
    """Represents a FEN string and its subsequent moves."""

    original_fen: fen
    subsequent_moves: list[chess.Move] = field(
        default_factory=lambda: list[chess.Move]()
    )


@dataclass
class FenPlusMoveHistory:
    """Represents a FEN string and its move history."""

    current_fen: fen
    historical_moves: list[moveUci] = field(default_factory=lambda: list[moveUci]())


@dataclass
class FenPlusHistory:
    """Represents a FEN string and its move history, along with historical board states."""

    current_fen: fen
    historical_moves: list[moveUci] = field(default_factory=lambda: list[moveUci]())
    historical_boards: list[chess._BoardState] = field(  # pyright: ignore[reportPrivateUsage]
        default_factory=lambda: list[chess._BoardState]()  # pyright: ignore[reportPrivateUsage]
    )

    def current_turn(self) -> chess.Color:
        """Returns the color of the player to move."""
        # copy of some code in the chess python library that cannot be easily extracted or called directly
        parts = self.current_fen.split()

        # Board part.
        try:
            _ = parts.pop(0)
        except IndexError:
            raise ValueError("empty fen") from None

        # Turn.
        try:
            turn_part = parts.pop(0)
        except IndexError:
            turn = chess.WHITE
        else:
            if turn_part == "w":
                turn = chess.WHITE
            elif turn_part == "b":
                turn = chess.BLACK
            else:
                raise ValueError(f"expected 'w' or 'b' for turn part of fen: {fen!r}")
        return turn


def square_rotate(square: chess.Square) -> chess.Square:
    """Rotates the square 180."""
    return square ^ 0x3F


def bitboard_rotate(bitboard: chess.Bitboard) -> chess.Bitboard:
    """Rotates the square 180."""
    return chess.flip_horizontal(bb=chess.flip_vertical(bb=bitboard))
