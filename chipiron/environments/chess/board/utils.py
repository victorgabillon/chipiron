import typing
from dataclasses import dataclass, field

import chess

from chipiron.environments.chess.move import moveUci

fen = typing.Annotated[str, "a string representing a fen"]


@dataclass
class FenPlusMoves:
    original_fen: fen
    subsequent_moves: list[chess.Move] = field(default_factory=list)


@dataclass
class FenPlusMoveHistory:
    current_fen: fen
    historical_moves: list[moveUci] = field(default_factory=list)


@dataclass
class FenPlusHistory:
    current_fen: fen
    historical_moves: list[moveUci] = field(default_factory=list)
    historical_boards: list[chess._BoardState] = field(default_factory=list)


def square_rotate(square: chess.Square) -> chess.Square:
    """Rotates the square 180."""
    return square ^ 0x3F


def bitboard_rotate(bitboard: chess.Bitboard) -> chess.Bitboard:
    """Rotates the square 180."""
    return chess.flip_horizontal(bb=chess.flip_vertical(bb=bitboard))
