"""Module for the (game-agnostic) Player shell.

The core `Player` is generic and delegates all game-specific logic to a `GameAdapter`:
- snapshot -> runtime state
- legal action enumeration
- optional oracle fast-path (e.g. syzygy in chess)
"""

from typing import Protocol, TypeVar

from valanga.game import BranchName, Seed
from valanga.policy import NotifyProgressCallable, Recommendation

PlayerId = str

StateSnapT = TypeVar("StateSnapT", contravariant=True)
RuntimeStateT = TypeVar("RuntimeStateT")


class PlayerMoveSelectionError(ValueError):
    """Raised when a move cannot be selected for a player."""

    def __init__(self) -> None:
        super().__init__("No legal moves in this position")


class GameAdapter(Protocol[StateSnapT, RuntimeStateT]):
    """Game-specific behavior injected into the generic `Player`."""

    def build_runtime_state(self, snapshot: StateSnapT) -> RuntimeStateT:
        """Build runtime state."""
        ...

    def legal_action_count(self, runtime_state: RuntimeStateT) -> int:
        """Legal action count."""
        ...

    def only_action_name(self, runtime_state: RuntimeStateT) -> BranchName:
        """Only action name."""
        ...

    def oracle_action_name(self, runtime_state: RuntimeStateT) -> BranchName | None:
        """Oracle action name."""
        ...

    def recommend(
        self,
        runtime_state: RuntimeStateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend."""
        ...


class Player[StateSnapT, RuntimeStateT]:
    """Fully game-agnostic player: recommends an action given a snapshot."""

    id: PlayerId
    adapter: GameAdapter[StateSnapT, RuntimeStateT]

    def __init__(
        self, name: str, adapter: GameAdapter[StateSnapT, RuntimeStateT]
    ) -> None:
        """Initialize the instance."""
        self.id = name
        self.adapter = adapter

    def get_id(self) -> PlayerId:
        """Return id."""
        return self.id

    def select_move(
        self,
        state_snapshot: StateSnapT,
        seed: Seed,
        notify_percent_function: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Select move."""
        runtime_state = self.adapter.build_runtime_state(state_snapshot)

        n = self.adapter.legal_action_count(runtime_state)
        if n == 0:
            raise PlayerMoveSelectionError

        # Fast path: if only one legal action, skip selection/search for humans.
        if n == 1 and self.id == "Human":
            only_name = self.adapter.only_action_name(runtime_state)
            return Recommendation(recommended_name=only_name)

        # Optional oracle fast-path (e.g., Syzygy). Return None if not applicable.
        oracle_name = self.adapter.oracle_action_name(runtime_state)
        if oracle_name is not None:
            return Recommendation(recommended_name=oracle_name)

        return self.adapter.recommend(
            runtime_state, seed, notify_progress=notify_percent_function
        )
