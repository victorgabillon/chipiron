from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable

if TYPE_CHECKING:
    import atomheart.board as boards
    from atomheart.board.utils import FenPlusHistory


class ChessBoardFactory(Protocol):
    def __call__(self, *, fen_with_history: "FenPlusHistory") -> "boards.IBoard": ...


@dataclass(frozen=True)
class ChessEnvironmentDeps:
    board_factory: ChessBoardFactory
    syzygy_table: AnySyzygyTable | None


@dataclass(frozen=True)
class CheckersEnvironmentDeps:
    pass
