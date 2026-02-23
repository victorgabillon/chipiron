"""Module for chess syzygy oracle."""

from typing import TYPE_CHECKING

from valanga.game import BranchName
from valanga.over_event import OverEvent

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable
from chipiron.players.oracles import PolicyOracle, TerminalOracle, ValueOracle

if TYPE_CHECKING:
    from atomheart.games.chess.move import MoveUci
    from atomheart.games.chess.move.imove import MoveKey


class ChessSyzygyPolicyOracle(PolicyOracle[ChessState]):
    """Policy oracle wrapper around Syzygy for chess-specific best-move queries."""

    def __init__(self, syzygy: AnySyzygyTable) -> None:
        """Initialize the instance."""
        self._syzygy = syzygy

    def supports(self, state: ChessState) -> bool:
        """Supports."""
        return self._syzygy.fast_in_table(state.board)

    def recommend(self, state: ChessState) -> BranchName:
        """Recommend."""
        best_move_key: MoveKey = self._syzygy.best_move(state.board)
        best_move_uci: MoveUci = state.board.get_uci_from_move_key(best_move_key)
        return best_move_uci


class ChessSyzygyValueOracle(ValueOracle[ChessState]):
    """Value oracle wrapper around Syzygy for chess-specific evaluations."""

    def __init__(self, syzygy: AnySyzygyTable) -> None:
        """Initialize the instance."""
        self._syzygy = syzygy

    def supports(self, state: ChessState) -> bool:
        """Supports."""
        return self._syzygy.fast_in_table(state.board)

    def value_white(self, state: ChessState) -> float:
        """Value white."""
        return float(self._syzygy.val(state.board))


class ChessSyzygyTerminalOracle(TerminalOracle[ChessState]):
    """Terminal oracle wrapper around Syzygy for chess-specific endgame metadata."""

    def __init__(self, syzygy: AnySyzygyTable) -> None:
        """Initialize the instance."""
        self._syzygy = syzygy

    def supports(self, state: ChessState) -> bool:
        """Supports."""
        return self._syzygy.fast_in_table(state.board)

    def over_event(self, state: ChessState) -> OverEvent:
        """Over event."""
        who_is_winner, how_over = self._syzygy.get_over_event(board=state.board)
        return OverEvent(
            how_over=how_over,
            who_is_winner=who_is_winner,
            termination=None,
        )
