"""Module for `GamePlayer`.

`GamePlayer` wraps a `Player` with game/match-specific metadata (e.g. color).
It stays generic and does not depend on any specific game snapshot/runtime types.
"""

from typing import Generic, TypeVar

from valanga.game import Color, Seed
from valanga.policy import Recommendation

from .player import Player

StateSnapT = TypeVar("StateSnapT")
RuntimeStateT = TypeVar("RuntimeStateT")


class GamePlayer(Generic[StateSnapT, RuntimeStateT]):
    """Wraps a `Player` for a specific game and color."""

    _player: Player[StateSnapT, RuntimeStateT]
    color: Color

    def __init__(self, player: Player[StateSnapT, RuntimeStateT], color: Color) -> None:
        self.color = color
        self._player = player

    @property
    def player(self) -> Player[StateSnapT, RuntimeStateT]:
        return self._player

    def select_move_from_snapshot(
        self, snapshot: StateSnapT, seed: Seed
    ) -> Recommendation:
        return self._player.select_move(state_snapshot=snapshot, seed=seed)
