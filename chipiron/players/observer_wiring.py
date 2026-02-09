"""Module for observer wiring."""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from chipiron.players.game_player import GamePlayer

SnapT_contra = TypeVar("SnapT_contra", contravariant=True)
RuntimeT = TypeVar("RuntimeT")
BuildArgsT = TypeVar("BuildArgsT")
BuildArgsT_contra = TypeVar("BuildArgsT_contra", contravariant=True)


class BuildGamePlayer(Protocol[SnapT_contra, RuntimeT, BuildArgsT_contra]):
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
