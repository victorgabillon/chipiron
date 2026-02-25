"""Module for creating player observers.

This module is game-agnostic orchestration:
- selects game-specific wiring by `GameKind`
- creates either a multi-process PlayerProcess or an in-process GamePlayer
- returns a `PlayerHandle` + a move-request function

Typing strategy:
- strong typing lives inside each wiring module
- a single cast occurs when selecting wiring by runtime `game_kind`
"""

import multiprocessing
from functools import partial
from typing import Protocol, assert_never, cast

from valanga import Color

from chipiron.environments.checkers.players.wiring.checkers_wiring import (
    CHECKERS_WIRING,
)
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_message import PlayerRequest
from chipiron.players.communications.player_runtime import handle_player_request
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.player_args import PlayerFactoryArgs
from chipiron.players.player_handle import InProcessPlayerHandle, PlayerHandle
from chipiron.players.player_thread import PlayerProcess
from chipiron.players.wirings.chess_wiring import CHESS_WIRING
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.queue_protocols import PutGetQueue, PutQueue


class MoveFunction(Protocol):
    """Protocol for functions that dispatch player requests."""

    def __call__(self, request: PlayerRequest[object]) -> None:
        """Dispatch a player request to its handler."""
        ...


class PlayerObserverFactory(Protocol):
    """Protocol for factories that build player observers."""

    def __call__(
        self,
        player_factory_args: PlayerFactoryArgs,
        player_color: Color,
        main_thread_mailbox: PutQueue[MainMailboxMessage],
    ) -> tuple[PlayerHandle, MoveFunction]:
        """Create a player observer and its request sender."""
        ...


def _select_wiring(game_kind: GameKind) -> ObserverWiring[object, object, object]:
    """Select the observer wiring for the requested game kind."""
    match game_kind:
        case GameKind.CHESS:
            return cast("ObserverWiring[object, object, object]", CHESS_WIRING)
        case GameKind.CHECKERS:
            return cast("ObserverWiring[object, object, object]", CHECKERS_WIRING)
        case _:
            assert_never(game_kind)


def get_observer_wiring_for_game_kind(
    game_kind: GameKind,
) -> ObserverWiring[object, object, object]:
    """Return the observer wiring for a game kind.

    Public helper to make wiring selection explicit and testable.
    """
    return _select_wiring(game_kind)


def create_player_observer_factory(
    *,
    game_kind: GameKind,
    each_player_has_its_own_thread: bool,
    implementation_args: object,
    universal_behavior: bool,
) -> PlayerObserverFactory:
    """Create player observer factory."""
    wiring = _select_wiring(game_kind)

    if each_player_has_its_own_thread:
        return cast(
            "PlayerObserverFactory",
            partial(
                _create_player_observer_distributed,
                wiring=wiring,
                implementation_args=implementation_args,
                universal_behavior=universal_behavior,
            ),
        )

    return cast(
        "PlayerObserverFactory",
        partial(
            _create_player_observer_mono_process,
            wiring=wiring,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        ),
    )


def _create_player_observer_distributed(
    player_factory_args: PlayerFactoryArgs,
    player_color: Color,
    main_thread_mailbox: PutQueue[IsDataclass],
    *,
    wiring: ObserverWiring[object, object, object],
    implementation_args: object,
    universal_behavior: bool,
) -> tuple[PlayerHandle, MoveFunction]:
    """Create a player observer backed by a separate process."""
    mgr = multiprocessing.Manager()
    player_process_mailbox: PutGetQueue[PlayerRequest[object] | None] = mgr.Queue()

    # `ObserverWiring.build_args_type` is game-specific; after type-erasure it must be
    # instantiated via a local cast.
    build_args_type = cast("type", wiring.build_args_type)
    build_args = build_args_type(
        player_factory_args=player_factory_args,
        player_color=player_color,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
    )

    proc: PlayerProcess[object, object, object] = PlayerProcess(
        build_game_player=wiring.build_game_player,
        build_args=build_args,
        queue_in=player_process_mailbox,
        queue_out=main_thread_mailbox,
    )
    proc.start()

    def send_request_to_mailbox(
        request: PlayerRequest[object],
        mailbox: PutQueue[PlayerRequest[object] | None],
    ) -> None:
        mailbox.put(request)

    move_function = partial(send_request_to_mailbox, mailbox=player_process_mailbox)
    return proc, cast("MoveFunction", move_function)


def _create_player_observer_mono_process(
    player_factory_args: PlayerFactoryArgs,
    player_color: Color,
    main_thread_mailbox: PutQueue[MainMailboxMessage],
    *,
    wiring: ObserverWiring[object, object, object],
    implementation_args: object,
    universal_behavior: bool,
) -> tuple[PlayerHandle, MoveFunction]:
    """Create a player observer running in-process."""
    build_args_type = cast("type", wiring.build_args_type)
    build_args = build_args_type(
        player_factory_args=player_factory_args,
        player_color=player_color,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
    )

    game_player = wiring.build_game_player(build_args)

    handle: PlayerHandle = InProcessPlayerHandle(game_player)

    def run_request_inline(
        request: PlayerRequest[object],
        *,
        queue_move: PutQueue[MainMailboxMessage],
    ) -> None:
        handle_player_request(
            request=request, game_player=game_player, out_queue=queue_move
        )

    move_function = partial(run_request_inline, queue_move=main_thread_mailbox)
    return handle, cast("MoveFunction", move_function)
