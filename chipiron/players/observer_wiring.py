from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from chipiron.players.game_player import GamePlayer
    from chipiron.utils.dataclass import IsDataclass
    from chipiron.utils.queue_protocols import PutQueue

SnapT = TypeVar("SnapT")
RuntimeT = TypeVar("RuntimeT")
BuildArgsT = TypeVar("BuildArgsT")
BuildArgsT_contra = TypeVar("BuildArgsT_contra", contravariant=True)


class BuildGamePlayer(Protocol[SnapT, RuntimeT, BuildArgsT_contra]):
    def __call__(
        self, args: BuildArgsT_contra, queue_out: PutQueue[IsDataclass]
    ) -> GamePlayer[SnapT, RuntimeT]: ...


@dataclass(frozen=True)
class ObserverWiring(Generic[SnapT, RuntimeT, BuildArgsT]):
    """Game-specific wiring used by the generic observer factory.

    This keeps chess/checkers-specific construction code out of the orchestration.
    """

    build_game_player: BuildGamePlayer[SnapT, RuntimeT, BuildArgsT]
    build_args_type: type[BuildArgsT]
