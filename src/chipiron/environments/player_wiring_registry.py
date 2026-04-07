"""Environment-level registry for game-specific observer wiring."""

# pylint: disable=import-outside-toplevel

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
            )  # pylint: disable=import-outside-toplevel

            return cast("ObserverWiring[object, object, object]", CHESS_WIRING)
        case GameKind.CHECKERS:
            from chipiron.environments.checkers.players.wiring.checkers_wiring import (
                CHECKERS_WIRING,
            )  # pylint: disable=import-outside-toplevel

            return cast("ObserverWiring[object, object, object]", CHECKERS_WIRING)
        case GameKind.INTEGER_REDUCTION:
            from chipiron.environments.integer_reduction.players.wiring.integer_reduction_wiring import (
                INTEGER_REDUCTION_WIRING,
            )  # pylint: disable=import-outside-toplevel

            return cast(
                "ObserverWiring[object, object, object]", INTEGER_REDUCTION_WIRING
            )
        case GameKind.MORPION:
            from chipiron.environments.morpion.players.wiring.morpion_wiring import (
                MORPION_WIRING,
            )  # pylint: disable=import-outside-toplevel

            return cast("ObserverWiring[object, object, object]", MORPION_WIRING)
        case _:
            raise UnsupportedGameKindError(game_kind)
