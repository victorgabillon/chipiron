"""Characterization tests for the current MatchController routing behavior."""

from __future__ import annotations

from test_support.runtime_fixtures import (
    build_runtime_harness,
    drain_gui_payloads,
    make_player_move,
)
from valanga import Color

from chipiron.displays.gui_protocol import (
    HumanActionChosen,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
    UpdStateGeneric,
)


def test_start_publishes_human_action_request_for_human_turn() -> None:
    """Freeze how the controller exposes a pending human action today."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )

    harness.controller.start()

    payloads = drain_gui_payloads(harness.display_queue)
    assert len(payloads) == 1
    payload = payloads[0]
    assert isinstance(payload, UpdNeedHumanAction)
    assert payload.ctx.request_id == 0
    assert payload.ctx.role_to_play is Color.WHITE
    assert payload.state_tag == 0
    assert harness.controller.pending_role is Color.WHITE
    assert harness.controller.pending_request_id == 0
    assert harness.engine_requests[Color.WHITE] == []
    assert harness.engine_requests[Color.BLACK] == []


def test_start_dispatches_engine_request_for_engine_turn() -> None:
    """Freeze the current engine-dispatch path for a non-human side to move."""
    harness = build_runtime_harness(
        start_turn=Color.BLACK,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )

    harness.controller.start()

    assert drain_gui_payloads(harness.display_queue) == []
    assert len(harness.engine_requests[Color.BLACK]) == 1
    request = harness.engine_requests[Color.BLACK][0]
    assert request.ctx is not None
    assert request.ctx.request_id == 0
    assert request.ctx.role_to_play is Color.BLACK
    assert request.state.current_state_tag == 0
    assert request.state.role_to_play is Color.BLACK


def test_handle_player_action_ignores_stale_request_id() -> None:
    """Freeze the controller's current stale-request rejection behavior."""
    harness = build_runtime_harness(
        start_turn=Color.BLACK,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )
    harness.controller.start()
    initial_request = harness.engine_requests[Color.BLACK][0]

    stale_move = make_player_move(initial_request, player_name="black-player")
    assert stale_move.ctx is not None
    stale_move = stale_move.__class__(
        branch_name=stale_move.branch_name,
        corresponding_state_tag=stale_move.corresponding_state_tag,
        ctx=stale_move.ctx.__class__(
            request_id=999,
            role_to_play=stale_move.ctx.role_to_play,
        ),
        player_name=stale_move.player_name,
        player_role=stale_move.player_role,
        evaluation=stale_move.evaluation,
    )

    harness.controller.handle_player_action(stale_move)

    assert harness.game_manager.game.state.tag == 0
    assert len(harness.engine_requests[Color.BLACK]) == 1
    assert drain_gui_payloads(harness.display_queue) == []


def test_handle_player_action_invalid_branch_reissues_same_request() -> None:
    """Freeze the current invalid-engine-move retry behavior."""
    harness = build_runtime_harness(
        start_turn=Color.BLACK,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )
    harness.controller.start()
    initial_request = harness.engine_requests[Color.BLACK][0]

    harness.controller.handle_player_action(
        make_player_move(
            initial_request,
            branch_name="illegal-branch",
            player_name="black-player",
        )
    )

    assert harness.game_manager.game.state.tag == 0
    assert len(harness.engine_requests[Color.BLACK]) == 2
    retried_request = harness.engine_requests[Color.BLACK][1]
    assert retried_request.ctx is not None
    assert initial_request.ctx is not None
    assert retried_request.ctx.request_id == initial_request.ctx.request_id
    assert drain_gui_payloads(harness.display_queue) == []


def test_handle_human_action_with_mismatched_state_tag_is_ignored() -> None:
    """Freeze the state-tag guard on human actions before later refactors."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )
    harness.controller.start()
    first_payload = drain_gui_payloads(harness.display_queue)[0]
    assert isinstance(first_payload, UpdNeedHumanAction)

    harness.controller.handle_human_action(
        HumanActionChosen(
            action_name="advance",
            ctx=first_payload.ctx,
            corresponding_state_tag=999,
        )
    )

    assert harness.game_manager.game.state.tag == 0
    assert drain_gui_payloads(harness.display_queue) == []
    assert harness.engine_requests[Color.BLACK] == []


def test_handle_human_action_applies_move_and_requests_next_actor() -> None:
    """Freeze the current update ordering after a valid human move."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=2,
        human_colors={Color.WHITE},
    )
    harness.controller.start()
    first_payload = drain_gui_payloads(harness.display_queue)[0]
    assert isinstance(first_payload, UpdNeedHumanAction)

    harness.controller.handle_human_action(
        HumanActionChosen(
            action_name="advance",
            ctx=first_payload.ctx,
            corresponding_state_tag=first_payload.state_tag,
        )
    )

    payloads = drain_gui_payloads(harness.display_queue)
    assert [type(payload) for payload in payloads] == [
        UpdNoHumanActionPending,
        UpdStateGeneric,
    ]
    assert harness.game_manager.game.state.turn is Color.BLACK
    assert harness.game_manager.game.state.tag == 1
    assert len(harness.engine_requests[Color.BLACK]) == 1
    next_request = harness.engine_requests[Color.BLACK][0]
    assert next_request.ctx is not None
    assert next_request.ctx.request_id == 1
    assert next_request.ctx.role_to_play is Color.BLACK


def test_handle_human_action_on_terminal_move_publishes_double_final_updates() -> None:
    """Freeze the current duplicate final GUI notifications on terminal human moves.

    This captures today's behavior for the safety net and may intentionally
    change during the later generic-role refactor.
    """
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=1,
        human_colors={Color.WHITE},
    )
    harness.controller.start()
    first_payload = drain_gui_payloads(harness.display_queue)[0]
    assert isinstance(first_payload, UpdNeedHumanAction)

    harness.controller.handle_human_action(
        HumanActionChosen(
            action_name="advance",
            ctx=first_payload.ctx,
            corresponding_state_tag=first_payload.state_tag,
        )
    )

    payloads = drain_gui_payloads(harness.display_queue)
    assert [type(payload) for payload in payloads] == [
        UpdNoHumanActionPending,
        UpdStateGeneric,
        UpdNoHumanActionPending,
        UpdStateGeneric,
    ]
    assert harness.controller.pending_role is None
    assert harness.controller.pending_request_id is None
    assert harness.engine_requests[Color.BLACK] == []
    assert harness.game_manager.game.state.is_game_over() is True
