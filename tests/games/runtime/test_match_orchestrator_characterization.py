"""Characterization tests for the current MatchOrchestrator mailbox loop."""

from __future__ import annotations

from test_support.runtime_fixtures import (
    build_runtime_harness,
    make_back_command,
    make_human_action_command,
    make_status_command,
    start_orchestrator_thread,
    wait_for_gui_payload,
    wait_for_thread_result,
)

from valanga import Color

from chipiron.displays.gui_protocol import UpdGameStatus, UpdNeedHumanAction, UpdNoHumanActionPending
from chipiron.games.domain.game.game_playing_status import PlayingStatus


def test_play_one_game_ignores_stale_scope_before_valid_human_move() -> None:
    """Freeze stale-scope rejection in the mailbox loop."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=1,
        human_colors={Color.WHITE, Color.BLACK},
    )
    run = start_orchestrator_thread(harness)
    first_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)

    stale_scope = harness.scope.__class__(
        session_id=harness.scope.session_id,
        match_id=harness.scope.match_id,
        game_id="stale-game",
    )
    harness.mailbox.put(
        make_human_action_command(
            scope=stale_scope,
            action_name="advance",
            ctx=first_request.ctx,
            corresponding_state_tag=first_request.state_tag,
        )
    )
    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=first_request.ctx,
            corresponding_state_tag=first_request.state_tag,
        )
    )

    report = wait_for_thread_result(run)

    assert report.action_history == ["advance"]
    assert report.state_tag_history == [0, 1]
    assert harness.game_manager.game.state.tag == 1


def test_pause_invalidates_pending_request_and_play_reissues_current_turn() -> None:
    """Freeze pause/play request invalidation before the larger role refactor."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=1,
        human_colors={Color.WHITE, Color.BLACK},
    )
    run = start_orchestrator_thread(harness)
    first_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)

    harness.mailbox.put(
        make_status_command(scope=harness.scope, status=PlayingStatus.PAUSE)
    )
    pause_status = wait_for_gui_payload(harness.display_queue, UpdGameStatus)
    assert pause_status.status is PlayingStatus.PAUSE
    wait_for_gui_payload(harness.display_queue, UpdNoHumanActionPending)

    harness.mailbox.put(
        make_status_command(scope=harness.scope, status=PlayingStatus.PLAY)
    )
    play_status = wait_for_gui_payload(harness.display_queue, UpdGameStatus)
    assert play_status.status is PlayingStatus.PLAY
    replay_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)
    assert replay_request.ctx.request_id == first_request.ctx.request_id + 1

    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=first_request.ctx,
            corresponding_state_tag=first_request.state_tag,
        )
    )
    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=replay_request.ctx,
            corresponding_state_tag=replay_request.state_tag,
        )
    )

    report = wait_for_thread_result(run)

    assert report.action_history == ["advance"]
    assert report.state_tag_history == [0, 1]


def test_back_one_move_rewinds_state_but_keeps_history_entries() -> None:
    """Freeze the current rewind behavior, including its awkward retained history."""
    harness = build_runtime_harness(
        start_turn=Color.WHITE,
        remaining_moves=2,
        human_colors={Color.WHITE, Color.BLACK},
    )
    run = start_orchestrator_thread(harness)
    first_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)

    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=first_request.ctx,
            corresponding_state_tag=first_request.state_tag,
        )
    )
    second_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)
    assert second_request.ctx.request_id == 1
    assert harness.game_manager.game.state.turn is Color.BLACK

    harness.mailbox.put(make_back_command(scope=harness.scope))
    pause_status = wait_for_gui_payload(harness.display_queue, UpdGameStatus)
    assert pause_status.status is PlayingStatus.PAUSE
    wait_for_gui_payload(harness.display_queue, UpdNoHumanActionPending)

    assert harness.game_manager.game.state.turn is Color.WHITE
    assert harness.game_manager.game.state.tag == 0
    assert harness.game_manager.game.is_paused() is True

    harness.mailbox.put(
        make_status_command(scope=harness.scope, status=PlayingStatus.PLAY)
    )
    play_status = wait_for_gui_payload(harness.display_queue, UpdGameStatus)
    assert play_status.status is PlayingStatus.PLAY
    replay_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)
    assert replay_request.ctx.request_id == 2
    assert replay_request.state_tag == 0

    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=replay_request.ctx,
            corresponding_state_tag=replay_request.state_tag,
        )
    )
    final_request = wait_for_gui_payload(harness.display_queue, UpdNeedHumanAction)
    assert final_request.ctx.request_id == 3

    harness.mailbox.put(
        make_human_action_command(
            scope=harness.scope,
            action_name="advance",
            ctx=final_request.ctx,
            corresponding_state_tag=final_request.state_tag,
        )
    )

    report = wait_for_thread_result(run)

    assert report.action_history == ["advance", "advance", "advance"]
    assert report.state_tag_history == [0, 1, 1, 2]
    assert harness.game_manager.game.state.tag == 2
