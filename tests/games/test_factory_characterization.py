"""Characterization tests for current role-order scheduling and results behavior."""

from __future__ import annotations

import pytest
from valanga import SOLO, Color

from chipiron.environments.chess.starting_position_args import FenStartingPositionArgs
from chipiron.environments.integer_reduction.starting_position_args import (
    IntegerReductionValueStartingPositionArgs,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.final_game_result import GameReport, RoleOutcome
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_args_factory import GameArgsFactory
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.games.domain.match.match_role_schedule import (
    ParticipantRoleTopologyMismatchError,
    SoloMatchSchedule,
    SoloTopologyRequiresSoloScheduleError,
    TwoRoleMatchSchedule,
    TwoRoleTopologyRequiresTwoRoleScheduleError,
    UnsupportedRoleTopologyError,
    ValidatedMatchPlan,
    build_validated_match_plan,
)
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.utils.small_tools import unique_int_from_list

STANDARD_CHESS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def build_two_role_plan(
    schedule: TwoRoleMatchSchedule,
) -> ValidatedMatchPlan:
    """Build a validated 2-role plan for current chess-style tests."""
    return build_validated_match_plan(
        participant_ids=("player-one", "player-two"),
        environment_roles=(Color.WHITE, Color.BLACK),
        schedule=schedule,
    )


def build_solo_plan(schedule: SoloMatchSchedule) -> ValidatedMatchPlan:
    """Build a validated solo plan for current integer-reduction-style tests."""
    return build_validated_match_plan(
        participant_ids=("solo-player",),
        environment_roles=(SOLO,),
        schedule=schedule,
    )


def test_generate_game_args_assigns_first_and_second_roles_from_neutral_schedule() -> (
    None
):
    """Freeze current 2-role scheduling using ordered environment roles."""
    two_role_schedule = TwoRoleMatchSchedule(
        number_of_games_player_one_on_first_role=1,
        number_of_games_player_one_on_second_role=2,
    )
    factory = GameArgsFactory(
        args_player_one=PlayerArgs(
            name="player-one",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        args_player_two=PlayerArgs(
            name="player-two",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        seed_=11,
        args_game=GameArgs(
            game_kind=GameKind.CHESS,
            starting_position=FenStartingPositionArgs(fen=STANDARD_CHESS_FEN),
        ),
        match_plan=build_two_role_plan(two_role_schedule),
    )

    first_mapping, _, _ = factory.generate_game_args(0)
    second_mapping, _, _ = factory.generate_game_args(1)

    assert first_mapping[Color.WHITE].player_args.name == "player-one"
    assert first_mapping[Color.BLACK].player_args.name == "player-two"
    assert second_mapping[Color.WHITE].player_args.name == "player-two"
    assert second_mapping[Color.BLACK].player_args.name == "player-one"


def test_generate_game_args_merges_seed_and_tracks_match_completion() -> None:
    """Freeze the current per-game seed merge and completion counting behavior."""
    two_role_schedule = TwoRoleMatchSchedule(
        number_of_games_player_one_on_first_role=1,
        number_of_games_player_one_on_second_role=2,
    )
    factory = GameArgsFactory(
        args_player_one=PlayerArgs(
            name="player-one",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        args_player_two=PlayerArgs(
            name="player-two",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        seed_=17,
        args_game=GameArgs(
            game_kind=GameKind.CHESS,
            starting_position=FenStartingPositionArgs(fen=STANDARD_CHESS_FEN),
        ),
        match_plan=build_two_role_plan(two_role_schedule),
    )

    _, _, first_seed = factory.generate_game_args(0)
    _, _, second_seed = factory.generate_game_args(1)
    assert first_seed == unique_int_from_list([17, 0])
    assert second_seed == unique_int_from_list([17, 1])
    assert factory.is_match_finished() is False

    factory.generate_game_args(2)
    assert factory.is_match_finished() is True


def test_match_results_count_wins_by_participant_identity() -> None:
    """MatchResults should aggregate participant wins without color-specific inputs."""
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
    assert simple.wins_by_participant == {"player-one": 0, "player-two": 1}
    assert simple.draws == 0


def test_generate_game_args_supports_integer_reduction_solo_assignment() -> None:
    """Solo games should bind only the real solo role."""
    factory = GameArgsFactory(
        args_player_one=PlayerArgs(
            name="solo-player",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        args_player_two=None,
        seed_=5,
        args_game=GameArgs(
            game_kind=GameKind.INTEGER_REDUCTION,
            starting_position=IntegerReductionValueStartingPositionArgs(value=9),
        ),
        match_plan=build_solo_plan(SoloMatchSchedule(number_of_games=1)),
    )

    assignment, _, merged_seed = factory.generate_game_args(0)

    assert set(assignment) == {SOLO}
    assert assignment[SOLO].player_args.name == "solo-player"
    assert merged_seed == unique_int_from_list([5, 0])
    assert factory.is_match_finished() is True


def test_two_role_match_schedule_reports_total_games() -> None:
    """2-role schedules should expose their neutral total-game count."""
    schedule = TwoRoleMatchSchedule(
        number_of_games_player_one_on_first_role=3,
        number_of_games_player_one_on_second_role=4,
    )
    assert schedule.total_games == 7


def test_solo_match_schedule_reports_total_games() -> None:
    """Solo schedules should expose their neutral total-game count."""
    assert SoloMatchSchedule(number_of_games=3).total_games == 3


def test_build_validated_match_plan_rejects_two_role_schedule_for_solo_topology() -> (
    None
):
    """A solo environment should fail at validated-plan assembly time."""
    with pytest.raises(SoloTopologyRequiresSoloScheduleError):
        build_validated_match_plan(
            participant_ids=("solo-player",),
            environment_roles=(SOLO,),
            schedule=TwoRoleMatchSchedule(
                number_of_games_player_one_on_first_role=1,
                number_of_games_player_one_on_second_role=0,
            ),
        )


def test_build_validated_match_plan_rejects_solo_schedule_for_two_role_topology() -> (
    None
):
    """A 2-role environment should fail at validated-plan assembly time."""
    with pytest.raises(TwoRoleTopologyRequiresTwoRoleScheduleError):
        build_validated_match_plan(
            participant_ids=("player-one", "player-two"),
            environment_roles=(Color.WHITE, Color.BLACK),
            schedule=SoloMatchSchedule(number_of_games=1),
        )


def test_build_validated_match_plan_carries_ready_for_scheduler_data() -> None:
    """Validated plans should expose the final scheduler-facing contract."""
    plan = build_validated_match_plan(
        participant_ids=("player-one", "player-two"),
        environment_roles=(Color.WHITE, Color.BLACK),
        schedule=TwoRoleMatchSchedule(
            number_of_games_player_one_on_first_role=1,
            number_of_games_player_one_on_second_role=2,
        ),
    )

    assert plan.participant_ids == ("player-one", "player-two")
    assert plan.scheduled_roles == (Color.WHITE, Color.BLACK)
    assert plan.participant_count == 2
    assert plan.role_count == 2
    assert plan.is_two_role is True
    assert plan.requires_second_participant is True
    assert plan.first_role is Color.WHITE
    assert plan.second_role is Color.BLACK
    assert plan.role_participant_indexes(0) == ((Color.WHITE, 0), (Color.BLACK, 1))
    assert plan.role_participant_indexes(1) == ((Color.WHITE, 1), (Color.BLACK, 0))
    assert plan.total_games == 3


def test_build_validated_match_plan_exposes_solo_helpers() -> None:
    """Solo validated plans should expose one role and one participant cleanly."""
    plan = build_solo_plan(SoloMatchSchedule(number_of_games=2))

    assert plan.participant_ids == ("solo-player",)
    assert plan.participant_count == 1
    assert plan.role_count == 1
    assert plan.is_solo is True
    assert plan.requires_second_participant is False
    assert plan.solo_role is SOLO
    assert plan.role_participant_indexes(0) == ((SOLO, 0),)
    assert plan.total_games == 2


def test_build_validated_match_plan_rejects_role_participant_mismatch() -> None:
    """Configured participant count should match the declared environment roles."""
    with pytest.raises(ParticipantRoleTopologyMismatchError):
        build_validated_match_plan(
            participant_ids=("player-one",),
            environment_roles=(Color.WHITE, Color.BLACK),
            schedule=TwoRoleMatchSchedule(
                number_of_games_player_one_on_first_role=1,
                number_of_games_player_one_on_second_role=0,
            ),
        )

    with pytest.raises(ParticipantRoleTopologyMismatchError):
        build_validated_match_plan(
            participant_ids=("player-one", "player-two"),
            environment_roles=(SOLO,),
            schedule=SoloMatchSchedule(number_of_games=1),
        )


def test_build_validated_match_plan_rejects_three_role_environments() -> None:
    """Current match scheduling should fail clearly for unsupported 3-role games."""
    with pytest.raises(UnsupportedRoleTopologyError):
        build_validated_match_plan(
            participant_ids=("one", "two", "three"),
            environment_roles=("alpha", "beta", "gamma"),
            schedule=TwoRoleMatchSchedule(
                number_of_games_player_one_on_first_role=1,
                number_of_games_player_one_on_second_role=0,
            ),
        )
