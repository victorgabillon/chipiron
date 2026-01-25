from __future__ import annotations

from typing import TYPE_CHECKING

from chipiron.environments.chess.types import ChessState
from chipiron.players.oracle import Oracle

if TYPE_CHECKING:
    from atomheart.move import MoveUci
    from atomheart.move.imove import MoveKey
    from valanga.game import BranchName

    from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable


class ChessSyzygyOracle(Oracle[ChessState]):
    """Oracle wrapper around Syzygy for chess-specific best-move queries."""

    def __init__(self, syzygy: AnySyzygyTable) -> None:
        self._syzygy = syzygy

    def supports(self, state: ChessState) -> bool:
        return self._syzygy.fast_in_table(state.board)

    def recommend(self, state: ChessState) -> BranchName:
        best_move_key: MoveKey = self._syzygy.best_move(state.board)
        best_move_uci: MoveUci = state.get_uci_from_move_key(best_move_key)
        return best_move_uci
