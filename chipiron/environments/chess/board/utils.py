import typing
from dataclasses import dataclass, field

import chess

fen = typing.Annotated[str, 'fen']


@dataclass
class FenPlusMoves:
    original_fen: fen
    subsequent_moves: list[chess.Move] = field(default_factory=list)

@dataclass
class FenPlusMoveHistory:
    current_fen: fen
    historical_moves: list[chess.Move] = field(default_factory=list)

