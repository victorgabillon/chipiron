"""Module for player handle."""

from dataclasses import dataclass
from typing import Protocol

from chipiron.players.game_player import GamePlayer


class PlayerHandle(Protocol):
    """Minimal lifecycle interface for something that keeps a player alive.

    GameManager only needs to shut down players at the end of a game. Both
    in-process players and multiprocessing-based players can satisfy this.
    """

    def close(self) -> None:
        """Close."""
        ...

    def is_alive(self) -> bool:
        """Return whether alive."""
        ...


@dataclass(frozen=True)
class InProcessPlayerHandle[SnapT, RuntimeT]:
    """Inprocessplayerhandle implementation."""

    player: GamePlayer[SnapT, RuntimeT]

    def close(self) -> None:
        """Close."""
        return

    def is_alive(self) -> bool:
        """Return whether alive."""
        return True
