"""Integer reduction adapter."""

from valanga.game import BranchName, Seed
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.players.player import PlayerMoveSelectionError


class IntegerReductionAdapter:
    """Integer-reduction-specific adapter used by the game-agnostic `Player`."""

    def __init__(
        self,
        *,
        dynamics: IntegerReductionDynamics,
        main_move_selector: BranchSelector[IntegerReductionState],
    ) -> None:
        """Initialize the instance."""
        self._dynamics = dynamics
        self._main = main_move_selector

    def build_runtime_state(self, snapshot: int) -> IntegerReductionState:
        """Build runtime state."""
        return IntegerReductionState(value=snapshot, steps=0)

    def legal_action_count(self, runtime_state: IntegerReductionState) -> int:
        """Legal action count."""
        return len(self._dynamics.legal_actions(runtime_state).get_all())

    def only_action_name(self, runtime_state: IntegerReductionState) -> BranchName:
        """Only action name."""
        actions = self._dynamics.legal_actions(runtime_state).get_all()
        if len(actions) != 1:
            raise PlayerMoveSelectionError
        return self._dynamics.action_name(runtime_state, actions[0])

    def oracle_action_name(  # pylint: disable=useless-return
        self, runtime_state: IntegerReductionState
    ) -> BranchName | None:
        """Oracle action name."""
        del runtime_state
        return None

    def recommend(
        self,
        runtime_state: IntegerReductionState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend."""
        return self._main.recommend(
            state=runtime_state,
            seed=seed,
            notify_progress=notify_progress,
        )
