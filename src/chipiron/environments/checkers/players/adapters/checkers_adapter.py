"""Module for checkers adapter."""

from valanga.game import BranchName, Seed
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

from chipiron.environments.checkers.types import CheckersDynamics, CheckersState


class CheckersAdapterError(ValueError):
    """Base error for checkers adapter failures."""


class SingleLegalMoveRequiredError(CheckersAdapterError):
    """Raised when only_action_name is called with multiple legal moves."""

    def __init__(self, move_count: int) -> None:
        """Initialize the error with the legal move count."""
        super().__init__(
            f"only_action_name called but position has != 1 legal move (count={move_count})"
        )


class CheckersAdapter:
    """Checkers-specific adapter used by the game-agnostic `Player`."""

    def __init__(
        self,
        *,
        dynamics: CheckersDynamics,
        main_move_selector: BranchSelector[CheckersState],
    ) -> None:
        """Initialize the instance."""
        self._dynamics = dynamics
        self._main = main_move_selector

    def build_runtime_state(self, snapshot: str) -> CheckersState:
        """Build runtime state."""
        return CheckersState.from_text(snapshot)

    def legal_action_count(self, runtime_state: CheckersState) -> int:
        """Legal action count."""
        return len(self._dynamics.legal_actions(runtime_state).get_all())

    def only_action_name(self, runtime_state: CheckersState) -> BranchName:
        """Only action name."""
        actions = self._dynamics.legal_actions(runtime_state).get_all()
        if len(actions) != 1:
            raise SingleLegalMoveRequiredError(len(actions))
        return self._dynamics.action_name(runtime_state, actions[0])

    def oracle_action_name(self, runtime_state: CheckersState) -> BranchName | None:
        """Oracle action name."""

    def recommend(
        self,
        runtime_state: CheckersState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend."""
        return self._main.recommend(
            state=runtime_state,
            seed=seed,
            notify_progress=notify_progress,
        )
