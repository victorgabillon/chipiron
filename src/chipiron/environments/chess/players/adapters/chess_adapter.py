"""Module for chess adapter."""

from typing import TYPE_CHECKING

from atomheart.games.chess.board import BoardFactory
from atomheart.games.chess.board.utils import FenPlusHistory
from valanga.game import BranchName, Seed
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

from chipiron.core.oracles import PolicyOracle
from chipiron.environments.chess.types import ChessState
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from atomheart.games.chess.move import MoveUci
    from atomheart.games.chess.move.imove import MoveKey


class ChessAdapterError(ValueError):
    """Base error for chess adapter failures."""


class SingleLegalMoveRequiredError(ChessAdapterError):
    """Raised when only_action_name is called with multiple legal moves."""

    def __init__(self, move_count: int) -> None:
        """Initialize the error with the legal move count."""
        super().__init__(
            f"only_action_name called but position has != 1 legal move (count={move_count})"
        )


class ChessAdapter:
    """Chess-specific adapter used by the game-agnostic `Player`.

    - Snapshot: `FenPlusHistory` (picklable IPC payload)
    - Runtime: `IBoard`
    - Optional oracle: Syzygy
    """

    def __init__(
        self,
        *,
        board_factory: BoardFactory,
        main_move_selector: BranchSelector[ChessState],
        oracle: PolicyOracle[ChessState] | None,
    ) -> None:
        """Initialize the instance."""
        self.board_factory = board_factory
        self.main_move_selector = main_move_selector
        self.oracle = oracle

    def build_runtime_state(self, snapshot: FenPlusHistory) -> ChessState:
        """Build runtime state."""
        return ChessState(self.board_factory(fen_with_history=snapshot))

    def legal_action_count(self, runtime_state: ChessState) -> int:
        # Keep the API generic; for chess we count legal moves.
        """Legal action count."""
        return len(runtime_state.board.legal_moves.get_all())

    def only_action_name(self, runtime_state: ChessState) -> BranchName:
        """Only action name."""
        keys = runtime_state.board.legal_moves.get_all()
        if len(keys) != 1:
            raise SingleLegalMoveRequiredError(len(keys))
        move_key: MoveKey = keys[0]
        move_uci: MoveUci = runtime_state.board.get_uci_from_move_key(move_key)
        return move_uci

    def oracle_action_name(self, runtime_state: ChessState) -> BranchName | None:
        """Oracle action name."""
        if self.oracle is None:
            return None

        if self.oracle.supports(runtime_state):
            chipiron_logger.info("Playing with oracle")
            return self.oracle.recommend(runtime_state)

        return None

    def recommend(
        self,
        runtime_state: ChessState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        # `valanga.policy.BranchSelector` is `recommend(state, seed)`.
        """Recommend."""
        return self.main_move_selector.recommend(
            state=runtime_state, seed=seed, notify_progress=notify_progress
        )
