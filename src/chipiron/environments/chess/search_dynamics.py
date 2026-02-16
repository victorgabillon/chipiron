"""Depth-aware chess search dynamics."""

from dataclasses import dataclass
from typing import Any

from anemone.dynamics import SearchDynamics
from atomheart.move.imove import MoveKey
from valanga import Transition

from chipiron.environments.chess.types import ChessState


@dataclass(frozen=True)
class ChessSearchDynamics(SearchDynamics[ChessState]):
    """Search dynamics with depth-aware board-copy policy."""

    __anemone_search_dynamics__ = True

    copy_stack_until_depth: int = 2
    deep_copy_legal_moves: bool = True

    def legal_actions(self, state: ChessState) -> Any:
        """Return legal action collection for ``state``."""
        return state.board.legal_moves

    def action_name(self, state: ChessState, action: MoveKey) -> str:
        """Convert an action key to a stable branch name."""
        return state.board.get_uci_from_move_key(move_key=action)

    def action_from_name(self, state: ChessState, name: str) -> MoveKey:
        """Convert a stable branch name to an action key."""
        return state.board.get_move_key_from_uci(move_uci=name)

    def step(
        self,
        state: ChessState,
        action: MoveKey,
        *,
        depth: int = 0,
    ) -> Transition[ChessState]:
        """Apply ``action`` and return a transition using depth-aware copy behavior."""
        copy_stack = depth < self.copy_stack_until_depth
        board_copy = state.board.copy(
            stack=copy_stack,
            deep_copy_legal_moves=self.deep_copy_legal_moves,
        )
        _ = board_copy.play_move_key(move=action)
        next_state = ChessState(board_copy)
        return Transition(
            next_state=next_state,
            modifications=None,
            is_over=board_copy.is_game_over(),
            over_event=None,
            info={},
        )
