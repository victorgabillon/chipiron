"""Module for the (game-agnostic) Player shell.

The core `Player` is generic and delegates all game-specific logic to a `GameAdapter`:
- snapshot -> runtime state
- legal action enumeration
- optional oracle fast-path (e.g. syzygy in chess)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from valanga.policy import Recommendation

if TYPE_CHECKING:
    from valanga.game import BranchName, Seed

PlayerId = str

StateSnapT = TypeVar("StateSnapT", contravariant=True)
RuntimeStateT = TypeVar("RuntimeStateT")


class GameAdapter(Protocol[StateSnapT, RuntimeStateT]):
    """Game-specific behavior injected into the generic `Player`."""

    def build_runtime_state(self, snapshot: StateSnapT) -> RuntimeStateT: ...

    def legal_action_count(self, runtime_state: RuntimeStateT) -> int: ...

    def only_action_name(self, runtime_state: RuntimeStateT) -> BranchName: ...

    def oracle_action_name(self, runtime_state: RuntimeStateT) -> BranchName | None: ...

    def recommend(self, runtime_state: RuntimeStateT, seed: Seed) -> Recommendation: ...


class Player(Generic[StateSnapT, RuntimeStateT]):
    """Fully game-agnostic player: recommends an action given a snapshot."""

    id: PlayerId
    adapter: GameAdapter[StateSnapT, RuntimeStateT]

    def __init__(
        self, name: str, adapter: GameAdapter[StateSnapT, RuntimeStateT]
    ) -> None:
        self.id = name
        self.adapter = adapter

    def get_id(self) -> PlayerId:
        return self.id

    def select_move(self, snapshot: StateSnapT, seed: Seed) -> Recommendation:
        runtime_state = self.adapter.build_runtime_state(snapshot)

        n = self.adapter.legal_action_count(runtime_state)
        if n == 0:
            raise ValueError("No legal moves in this position")

        # Fast path: if only one legal action, skip selection/search for humans.
        if n == 1 and self.id == "Human":
            only_name = self.adapter.only_action_name(runtime_state)
            return Recommendation(recommended_name=only_name)

        # Optional oracle fast-path (e.g., Syzygy). Return None if not applicable.
        oracle_name = self.adapter.oracle_action_name(runtime_state)
        if oracle_name is not None:
            return Recommendation(recommended_name=oracle_name)

        return self.adapter.recommend(runtime_state, seed)
