"""Environment-level registry for game-specific observer wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from chipiron.environments.types import GameKind

if TYPE_CHECKING:
    from chipiron.players.observer_wiring import ObserverWiring


class UnsupportedGameKindError(ValueError):
    """Raised when no observer wiring exists for a game kind."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the unsupported game kind."""
        super().__init__(f"Unsupported game kind: {game_kind}")


def get_observer_wiring(game_kind: GameKind) -> ObserverWiring[object, object, object]:
    """Return observer wiring for the given game kind."""
    match game_kind:
        case GameKind.CHESS:
            from chipiron.environments.chess.players.wiring.chess_wiring import (
                CHESS_WIRING,
            )

            return cast("ObserverWiring[object, object, object]", CHESS_WIRING)
        case GameKind.CHECKERS:
            from chipiron.environments.checkers.players.wiring.checkers_wiring import (
                CHECKERS_WIRING,
            )

            return cast("ObserverWiring[object, object, object]", CHECKERS_WIRING)
        case _:
            raise UnsupportedGameKindError(game_kind)
