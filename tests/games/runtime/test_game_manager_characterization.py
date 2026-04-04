"""Characterization tests for the current GameManager synchronous behavior."""

from __future__ import annotations

from test_support.runtime_fixtures import build_runtime_harness

from valanga import Color

from chipiron.displays.gui_protocol import Scope
from chipiron.games.runtime.orchestrator.domain_events import (
    ActionApplied,
    IllegalAction,
    MatchOver,
    NeedAction,
)


def test_start_match_sync_requests_current_turn_with_request_id_zero() -> None:
    """Freeze the current initial request contract emitted by GameManager."""
    harness = build_runtime_harness(start_turn=Color.BLACK, remaining_moves=2)

    events = harness.game_manager.start_match_sync(harness.scope)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, NeedAction)
    assert event.scope == harness.scope
    assert event.color is Color.BLACK
    assert event.request_id == 0
    assert event.state is harness.game_manager.game.state


def test_propose_action_sync_returns_no_events_for_wrong_scope_stale_id_or_wrong_turn() -> None:
    """Freeze the current silent rejection behavior for mismatched requests."""
    harness = build_runtime_harness(start_turn=Color.WHITE, remaining_moves=2)
    harness.game_manager.start_match_sync(harness.scope)

    stale_scope = Scope(
        session_id=harness.scope.session_id,
        match_id=harness.scope.match_id,
        game_id="other-game",
    )

    assert (
        harness.game_manager.propose_action_sync(
            scope=stale_scope,
            color=Color.WHITE,
            request_id=0,
            action="advance",
        )
        == []
    )
    assert (
        harness.game_manager.propose_action_sync(
            scope=harness.scope,
            color=Color.WHITE,
            request_id=9,
            action="advance",
        )
        == []
    )
    assert (
        harness.game_manager.propose_action_sync(
            scope=harness.scope,
            color=Color.BLACK,
            request_id=0,
            action="advance",
        )
        == []
    )

    assert harness.game_manager.game.state.tag == 0
    assert harness.game_manager.game.ply == 0


def test_propose_action_sync_invalid_action_reissues_same_request() -> None:
    """Freeze the current invalid-action behavior before any refactor changes it."""
    harness = build_runtime_harness(start_turn=Color.WHITE, remaining_moves=2)
    harness.game_manager.start_match_sync(harness.scope)

    events = harness.game_manager.propose_action_sync(
        scope=harness.scope,
        color=Color.WHITE,
        request_id=0,
        action="not-a-real-action",
    )

    assert len(events) == 2
    assert isinstance(events[0], IllegalAction)
    assert isinstance(events[1], NeedAction)
    assert events[0].request_id == 0
    assert events[1].request_id == 0
    assert events[1].color is Color.WHITE
    assert harness.game_manager.game.state.tag == 0
    assert harness.game_manager.game.ply == 0


def test_propose_action_sync_valid_action_applies_transition_and_requests_next_turn() -> None:
    """Freeze the accepted-action event ordering and next-turn request behavior."""
    harness = build_runtime_harness(start_turn=Color.WHITE, remaining_moves=2)
    harness.game_manager.start_match_sync(harness.scope)

    events = harness.game_manager.propose_action_sync(
        scope=harness.scope,
        color=Color.WHITE,
        request_id=0,
        action="advance",
    )

    assert len(events) == 2
    assert isinstance(events[0], ActionApplied)
    assert isinstance(events[1], NeedAction)
    assert events[0].request_id == 0
    assert events[1].request_id == 1
    assert events[1].color is Color.BLACK
    assert harness.game_manager.game.state.turn is Color.BLACK
    assert harness.game_manager.game.state.tag == 1
    assert harness.game_manager.game.ply == 1
    assert harness.game_manager.game.action_history == ["advance"]


def test_propose_action_sync_terminal_transition_emits_match_over_without_next_request() -> None:
    """Freeze the current terminal-event shape emitted by GameManager."""
    harness = build_runtime_harness(start_turn=Color.WHITE, remaining_moves=1)
    harness.game_manager.start_match_sync(harness.scope)

    events = harness.game_manager.propose_action_sync(
        scope=harness.scope,
        color=Color.WHITE,
        request_id=0,
        action="advance",
    )

    assert len(events) == 2
    assert isinstance(events[0], ActionApplied)
    assert isinstance(events[1], MatchOver)
    assert harness.game_manager.game.state.is_game_over() is True
    assert harness.game_manager.game.state.last_actor is Color.WHITE
