"""Exercise native annotation parsing and real default-context player processes."""

import multiprocessing
import sys
from dataclasses import is_dataclass
from typing import get_type_hints

import pytest
from parsley import make_partial_dataclass_with_optional_paths
from valanga import SOLO

from chipiron.displays.gui_protocol import Scope
from chipiron.environments.integer_reduction.players.wiring.integer_reduction_wiring import (
    BuildIntegerReductionGamePlayerArgs,
    build_integer_reduction_game_player,
)
from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.players.communications.player_message import (
    EvMove,
    PlayerEvent,
    PlayerRequest,
    TurnStatePlusHistory,
)
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.players.player_args import PlayerArgs, PlayerFactoryArgs
from chipiron.players.player_thread import PlayerProcess
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.one_match.one_match import MatchScriptArgs


@pytest.mark.fast
def test_native_match_configuration_annotations_remain_resolvable() -> None:
    """Parse every nested argument type without mutating the caller's dataclass."""
    before = get_type_hints(MatchScriptArgs)
    partial = make_partial_dataclass_with_optional_paths(MatchScriptArgs)
    assert is_dataclass(partial)
    assert partial is not MatchScriptArgs
    assert get_type_hints(MatchScriptArgs) == before


@pytest.mark.fast
@pytest.mark.skipif(sys.platform != "linux", reason="Linux default-context check")
def test_real_player_uses_python314_default_process_context() -> None:
    """Start a real player, exchange pickled events, and shut down under forkserver."""
    assert multiprocessing.get_start_method() == "forkserver"
    context = multiprocessing.get_context()
    incoming = context.Queue()
    outgoing = context.Queue()
    state = IntegerReductionState(value=10)
    dynamics = IntegerReductionDynamics()
    legal_names = {
        dynamics.action_name(state, action) for action in dynamics.legal_actions(state)
    }
    scope = Scope(session_id="py314", match_id="match", game_id="game")
    player = PlayerProcess(
        build_game_player=build_integer_reduction_game_player,
        build_args=BuildIntegerReductionGamePlayerArgs(
            player_factory_args=PlayerFactoryArgs(
                player_args=PlayerArgs(
                    name="Random",
                    main_move_selector=RandomSelectorArgs(),
                    oracle_play=False,
                ),
                seed=17,
            ),
            player_role=SOLO,
            implementation_args=ImplementationArgs(),
            universal_behavior=True,
        ),
        queue_in=incoming,
        queue_out=outgoing,
    )
    player.start()
    try:
        incoming.put(
            PlayerRequest(
                schema_version=1,
                scope=scope,
                seed=17,
                state=TurnStatePlusHistory(
                    current_state_tag=state.tag,
                    role_to_play=SOLO,
                    snapshot=10,
                ),
            )
        )
        event = outgoing.get(timeout=30)
        assert isinstance(event, PlayerEvent)
        assert event.scope == scope
        assert isinstance(event.payload, EvMove)
        assert event.payload.branch_name in legal_names
        assert event.payload.player_role == SOLO
        player.stop()
        player.join(timeout=10)
        assert not player.is_alive()
        assert player.exitcode == 0
    finally:
        if player.is_alive():
            player.terminate()
            player.join(timeout=10)
        incoming.close()
        outgoing.close()
        incoming.join_thread()
        outgoing.join_thread()
