"""Module for observer wiring."""
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from chipiron.players.game_player import GamePlayer
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.queue_protocols import PutQueue

SnapT = TypeVar("SnapT")
RuntimeT = TypeVar("RuntimeT")
BuildArgsT = TypeVar("BuildArgsT")
BuildArgsT_contra = TypeVar("BuildArgsT_contra", contravariant=True)


class BuildGamePlayer(Protocol[SnapT, RuntimeT, BuildArgsT_contra]):
    """Buildgameplayer implementation."""

    def __call__(
        self,
        args: BuildArgsT_contra,
        queue_out: PutQueue[MainMailboxMessage],
    ) -> GamePlayer[SnapT, RuntimeT]:
        """Invoke the callable instance."""
        ...
@dataclass(frozen=True)
class ObserverWiring(Generic[SnapT, RuntimeT, BuildArgsT]):
    """Game-specific wiring used by the generic observer factory.

    This keeps chess/checkers-specific construction code out of the orchestration.
    """

    build_game_player: BuildGamePlayer[SnapT, RuntimeT, BuildArgsT]
    build_args_type: type[BuildArgsT]
