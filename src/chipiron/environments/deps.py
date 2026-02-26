"""Module for deps."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import (
    AnySyzygyTable,
)

if TYPE_CHECKING:
    import atomheart.games.chess.board as boards
    from atomheart.games.chess.board.utils import FenPlusHistory


class ChessBoardFactory(Protocol):
    """Protocol for building a chess board from FEN history."""

    def __call__(self, *, fen_with_history: "FenPlusHistory") -> "boards.IBoard":
        """Create a board from the provided FEN history."""
        ...


@dataclass(frozen=True)
class ChessEnvironmentDeps:
    """Dependencies required to build chess environments."""

    board_factory: ChessBoardFactory
    syzygy_table: AnySyzygyTable | None


@dataclass(frozen=True)
class CheckersEnvironmentDeps:
    """Dependencies required to build checkers environments."""

    forced_capture: bool = True


EnvDeps = ChessEnvironmentDeps | CheckersEnvironmentDeps
