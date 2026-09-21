"""Focused tests for PR5 role- and participant-based result reporting."""

from __future__ import annotations

from dataclasses import asdict

from chipiron.games.domain.game.final_game_result import (
    GameReport,
    RoleOutcome,
)
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.utils.dataclass import custom_asdict_factory


def test_match_results_keep_two_player_compatibility_while_aggregating_by_participant() -> (
    None
):
    """Two-player match summaries should be expressed through participant ordering only."""
    results = MatchResults(participant_ids=("player-one", "player-two"))

    results.add_result_one_game(
        game_report=GameReport(
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={"White": "player-two", "Black": "player-one"},
            result_by_role={"White": RoleOutcome.WIN, "Black": RoleOutcome.LOSS},
            winner_roles=["White"],
        )
    )

    simple = results.get_simple_result()
    assert simple.participant_order == ("player-one", "player-two")
    assert simple.wins_by_participant == {"player-one": 0, "player-two": 1}
    assert simple.draws == 0


def test_match_results_support_single_participant_games() -> None:
    """A solo participant should aggregate cleanly without fake opponents."""
    results = MatchResults(participant_ids=("solo-player",))

    results.add_result_one_game(
        game_report=GameReport(
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={"Solo": "solo-player"},
            result_by_role={"Solo": RoleOutcome.WIN},
            winner_roles=["Solo"],
        )
    )

    simple = results.get_simple_result()
    assert simple.participant_order == ("solo-player",)
    assert simple.stats_by_participant["solo-player"].wins == 1


def test_match_results_support_three_role_games() -> None:
    """Three-role aggregation should not assume exactly two participants."""
    results = MatchResults(participant_ids=("alpha", "beta", "gamma"))

    results.add_result_one_game(
        game_report=GameReport(
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={
                "North": "alpha",
                "East": "beta",
                "West": "gamma",
            },
            result_by_role={
                "North": RoleOutcome.WIN,
                "East": RoleOutcome.LOSS,
                "West": RoleOutcome.DRAW,
            },
            winner_roles=["North"],
        )
    )

    simple = results.get_simple_result()
    assert simple.participant_order == ("alpha", "beta", "gamma")
    assert simple.stats_by_participant["alpha"].wins == 1
    assert simple.stats_by_participant["beta"].losses == 1
    assert simple.stats_by_participant["gamma"].draws == 1


def test_game_report_serialization_is_role_based() -> None:
    """Serialized game reports should expose role and participant mappings explicitly."""
    report = GameReport(
        action_history=["move-a", "move-b"],
        state_tag_history=["state-0", "state-1"],
        participant_id_by_role={"White": "alice", "Black": "bob"},
        result_by_role={"White": RoleOutcome.DRAW, "Black": RoleOutcome.DRAW},
        winner_roles=[],
        result_reason="threefold_repetition",
    )

    payload = asdict(report, dict_factory=custom_asdict_factory)

    assert payload["participant_id_by_role"] == {"White": "alice", "Black": "bob"}
    assert payload["result_by_role"] == {"White": "draw", "Black": "draw"}
    assert payload["result_reason"] == "threefold_repetition"


def test_match_results_aggregate_multiple_games_per_participant() -> None:
    """Participant stats should accumulate across multiple reported games."""
    results = MatchResults(participant_ids=("alice", "bob"))

    results.add_result_one_game(
        game_report=GameReport(
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={"White": "alice", "Black": "bob"},
            result_by_role={"White": RoleOutcome.WIN, "Black": RoleOutcome.LOSS},
            winner_roles=["White"],
        )
    )
    results.add_result_one_game(
        game_report=GameReport(
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={"White": "bob", "Black": "alice"},
            result_by_role={"White": RoleOutcome.DRAW, "Black": RoleOutcome.DRAW},
            winner_roles=[],
        )
    )

    simple = results.get_simple_result()
    assert simple.games_played == 2
    assert simple.draws == 1
    assert simple.stats_by_participant["alice"].wins == 1
    assert simple.stats_by_participant["alice"].draws == 1
    assert simple.stats_by_participant["bob"].losses == 1
    assert simple.stats_by_participant["bob"].draws == 1
