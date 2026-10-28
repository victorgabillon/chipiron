"""Module for observer wiring."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from chipiron.players.game_player import GamePlayer


class BuildGamePlayer[SnapT_contra, RuntimeT, BuildArgsT_contra](Protocol):
    """Buildgameplayer implementation."""

    def __call__(
        self,
        args: BuildArgsT_contra,
    ) -> GamePlayer[SnapT_contra, RuntimeT]:
        """Invoke the callable instance."""
        ...


@dataclass(frozen=True)
class ObserverWiring[SnapT_contra, RuntimeT, BuildArgsT]:
    """Game-specific wiring used by the generic observer factory.

    This keeps chess/checkers-specific construction code out of the orchestration.
    """

    build_game_player: BuildGamePlayer[SnapT_contra, RuntimeT, BuildArgsT]
    build_args_type: type[BuildArgsT]
