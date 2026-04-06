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
from chipiron.games.domain.match.match_factories import (
    ParticipantRoleTopologyMismatchError,
    UnsupportedRoleTopologyError,
    validate_supported_match_topology,
)
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.games.domain.match.match_role_schedule import (
    SoloMatchSchedule,
    SoloTopologyRequiresSoloScheduleError,
    TwoRoleMatchSchedule,
    TwoRoleTopologyRequiresTwoRoleScheduleError,
)
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.utils.small_tools import unique_int_from_list

STANDARD_CHESS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_generate_game_args_assigns_first_and_second_roles_from_neutral_schedule() -> None:
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
        scheduled_roles=(Color.WHITE, Color.BLACK),
        schedule=two_role_schedule,
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
        scheduled_roles=(Color.WHITE, Color.BLACK),
        schedule=two_role_schedule,
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
        scheduled_roles=(SOLO,),
        schedule=SoloMatchSchedule(number_of_games=1),
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


def test_game_args_factory_rejects_two_role_schedule_for_solo_topology() -> None:
    """A solo environment should not accept a 2-role schedule."""
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
        scheduled_roles=(SOLO,),
        schedule=TwoRoleMatchSchedule(
            number_of_games_player_one_on_first_role=1,
            number_of_games_player_one_on_second_role=0,
        ),
    )

    with pytest.raises(SoloTopologyRequiresSoloScheduleError):
        factory.generate_game_args(0)


def test_game_args_factory_rejects_solo_schedule_for_two_role_topology() -> None:
    """A 2-role environment should not accept a solo schedule."""
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
        scheduled_roles=(Color.WHITE, Color.BLACK),
        schedule=SoloMatchSchedule(number_of_games=1),
    )

    with pytest.raises(TwoRoleTopologyRequiresTwoRoleScheduleError):
        factory.generate_game_args(0)


def test_validate_supported_match_topology_rejects_role_participant_mismatch() -> None:
    """Configured participant count should match the declared environment roles."""
    with pytest.raises(ParticipantRoleTopologyMismatchError):
        validate_supported_match_topology(
            participant_ids=("player-one",),
            environment_roles=(Color.WHITE, Color.BLACK),
        )

    with pytest.raises(ParticipantRoleTopologyMismatchError):
        validate_supported_match_topology(
            participant_ids=("player-one", "player-two"),
            environment_roles=(SOLO,),
        )


def test_validate_supported_match_topology_rejects_three_role_environments() -> None:
    """Current match scheduling should fail clearly for unsupported 3-role games."""
    with pytest.raises(UnsupportedRoleTopologyError):
        validate_supported_match_topology(
            participant_ids=("one", "two", "three"),
            environment_roles=("alpha", "beta", "gamma"),
        )
