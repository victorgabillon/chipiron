"""Module for `GamePlayer`.

`GamePlayer` wraps a `Player` with game/match-specific metadata (e.g. role).
It stays generic and does not depend on any specific game snapshot/runtime types.
"""

from typing import TypeVar

from valanga.game import Seed
from valanga.policy import NotifyProgressCallable, Recommendation

from chipiron.core.roles import GameRole

from .player import Player

StateSnapT = TypeVar("StateSnapT")
RuntimeStateT = TypeVar("RuntimeStateT")


class GamePlayer[StateSnapT, RuntimeStateT]:
    """Wraps a `Player` for a specific game role."""

    _player: Player[StateSnapT, RuntimeStateT]
    role: GameRole

    def __init__(self, player: Player[StateSnapT, RuntimeStateT], role: GameRole) -> None:
        """Initialize the instance."""
        self.role = role
        self._player = player

    @property
    def player(self) -> Player[StateSnapT, RuntimeStateT]:
        """Player."""
        return self._player

    def select_move_from_snapshot(
        self,
        snapshot: StateSnapT,
        seed: Seed,
        notify_percent_function: NotifyProgressCallable,
    ) -> Recommendation:
        """Select move from snapshot."""
        return self._player.select_move(
            state_snapshot=snapshot,
            seed=seed,
            notify_percent_function=notify_percent_function,
        )
