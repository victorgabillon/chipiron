"""Module for chess syzygy oracle."""

from typing import TYPE_CHECKING

from valanga import Color
from valanga.evaluations import Value
from valanga.game import BranchName
from valanga.over_event import OverEvent

from chipiron.core.oracles import PolicyOracle, TerminalOracle, ValueOracle
from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import (
    AnySyzygyTable,
)
from chipiron.environments.chess.types import ChessState

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

    def evaluate(self, state: ChessState) -> Value:
        """Value white."""
        return self._syzygy.evaluate(state.board)


class ChessSyzygyTerminalOracle(TerminalOracle[ChessState, Color]):
    """Terminal oracle wrapper around Syzygy for chess-specific endgame metadata."""

    def __init__(self, syzygy: AnySyzygyTable) -> None:
        """Initialize the instance."""
        self._syzygy = syzygy

    def supports(self, state: ChessState) -> bool:
        """Supports."""
        return self._syzygy.fast_in_table(state.board)

    def over_event(self, state: ChessState) -> OverEvent[Color]:
        """Over event."""
        who_is_winner, how_over = self._syzygy.get_over_event(board=state.board)
        return OverEvent[Color](
            outcome=how_over,
            winner=who_is_winner,
            termination=None,
        )
