from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chipiron.games.game.game_manager import MainMailboxMessage
from chipiron.players.observer_wiring import ObserverWiring

if TYPE_CHECKING:
    from valanga import Color

    from chipiron.players.game_player import GamePlayer
    from chipiron.players.player_args import PlayerFactoryArgs
    from chipiron.utils.queue_protocols import PutQueue

# Placeholder types until checkers runtime exists
CheckersSnap = Any
CheckersRuntime = Any


@dataclass(frozen=True)
class BuildCheckersGamePlayerArgs:
    player_factory_args: PlayerFactoryArgs
    player_color: Color
    implementation_args: object
    universal_behavior: bool


def build_checkers_game_player(
    args: BuildCheckersGamePlayerArgs,
    queue_out: PutQueue[MainMailboxMessage],
) -> GamePlayer[CheckersSnap, CheckersRuntime]:
    raise NotImplementedError("checkers player building not wired yet")


CHECKERS_WIRING: ObserverWiring[
    CheckersSnap, CheckersRuntime, BuildCheckersGamePlayerArgs
] = ObserverWiring(
    build_game_player=build_checkers_game_player,
    build_args_type=BuildCheckersGamePlayerArgs,
)
