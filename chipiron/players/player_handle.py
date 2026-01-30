from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from chipiron.players.game_player import GamePlayer

SnapT = TypeVar("SnapT")
RuntimeT = TypeVar("RuntimeT")


class PlayerHandle(Protocol):
    """Minimal lifecycle interface for something that keeps a player alive.

    GameManager only needs to shut down players at the end of a game. Both
    in-process players and multiprocessing-based players can satisfy this.
    """

    def close(self) -> None: ...

    def is_alive(self) -> bool: ...


@dataclass(frozen=True)
class InProcessPlayerHandle(Generic[SnapT, RuntimeT]):
    player: GamePlayer[SnapT, RuntimeT]

    def close(self) -> None:
        return

    def is_alive(self) -> bool:
        return True
