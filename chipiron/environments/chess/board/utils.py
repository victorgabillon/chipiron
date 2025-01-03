import typing
from dataclasses import dataclass, field

import chess

from chipiron.environments.chess.move import moveUci

fen = typing.Annotated[str, 'a string representing a fen']


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
