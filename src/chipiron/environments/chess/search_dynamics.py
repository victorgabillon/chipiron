"""Depth-aware chess search dynamics wrappers."""

from dataclasses import dataclass
from typing import Any

from anemone.dynamics import SearchDynamics
from atomheart.move.imove import MoveKey
from valanga import Transition

from chipiron.environments.chess.types import ChessState


@dataclass(frozen=True)
class ChessCopyStackSearchDynamics(SearchDynamics[ChessState, MoveKey]):
    """Wrap search dynamics with depth-aware board-copy policy in ``step`` only."""

    __anemone_search_dynamics__ = True

    base: SearchDynamics[ChessState, MoveKey]
    copy_stack_until_depth: int = 2
    deep_copy_legal_moves: bool = True

    def __getattr__(self, name: str) -> Any:
        """Delegate non-overridden behavior to the wrapped search dynamics."""
        return getattr(self.base, name)

    def legal_actions(self, state: ChessState) -> Any:
        """Return legal action collection for ``state``."""
        return self.base.legal_actions(state)

    def action_name(self, state: ChessState, action: MoveKey) -> str:
        """Convert an action key to a stable branch name."""
        return self.base.action_name(state, action)

    def action_from_name(self, state: ChessState, name: str) -> MoveKey:
        """Convert a stable branch name to an action key."""
        return self.base.action_from_name(state, name)

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
