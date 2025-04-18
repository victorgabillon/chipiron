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

    def current_turn(self) -> chess.Color:
        # copy of some code in the chess python library that cannot be easily extracted or called directly
        parts = self.current_fen.split()

        # Board part.
        try:
            _ = parts.pop(0)
        except IndexError:
            raise ValueError("empty fen")

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
