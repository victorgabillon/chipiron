from __future__ import annotations

from typing import TYPE_CHECKING

from chipiron.environments.chess.types import ChessState
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from atomheart.board import BoardFactory, IBoard
    from atomheart.board.utils import FenPlusHistory
    from atomheart.move import MoveUci
    from atomheart.move.imove import MoveKey
    from valanga.game import BranchName, Seed
    from valanga.policy import BranchSelector, Recommendation

    from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable


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
        main_move_selector: BranchSelector,
        syzygy: AnySyzygyTable | None,
    ) -> None:
        self.board_factory = board_factory
        self.main_move_selector = main_move_selector
        self.syzygy = syzygy

    def build_runtime_state(self, snapshot: FenPlusHistory) -> IBoard:
        return self.board_factory(fen_with_history=snapshot)

    def legal_action_count(self, runtime_state: IBoard) -> int:
        # Keep the API generic; for chess we count legal moves.
        return len(runtime_state.legal_moves.get_all())

    def only_action_name(self, runtime_state: IBoard) -> BranchName:
        keys = runtime_state.legal_moves.get_all()
        if len(keys) != 1:
            raise ValueError("only_action_name called but position has != 1 legal move")
        move_key: MoveKey = keys[0]
        move_uci: MoveUci = runtime_state.get_uci_from_move_key(move_key)
        return move_uci

    def oracle_action_name(self, runtime_state: IBoard) -> BranchName | None:
        if self.syzygy is None:
            return None

        if self.syzygy.fast_in_table(runtime_state):
            chipiron_logger.info("Playing with Syzygy")
            best_move_key: MoveKey = self.syzygy.best_move(runtime_state)
            best_move_uci: MoveUci = runtime_state.get_uci_from_move_key(best_move_key)
            return best_move_uci

        return None

    def recommend(self, runtime_state: ChessState, seed: Seed) -> Recommendation:
        # `valanga.policy.BranchSelector` is `recommend(state, seed)`.
        return self.main_move_selector.recommend(state=runtime_state, seed=seed)
