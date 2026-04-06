"""Characterization tests for current white/black factory and result assumptions."""

from __future__ import annotations

from types import SimpleNamespace

from valanga import SOLO, Color

from chipiron.environments.types import GameKind
from chipiron.games.domain.game.final_game_result import GameReport, RoleOutcome
from chipiron.games.domain.game.game_args_factory import GameArgsFactory
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.utils.small_tools import unique_int_from_list


def test_generate_game_args_assigns_white_and_black_from_player_one_quota() -> None:
    """Freeze the current white/black assignment logic for match scheduling."""
    factory = GameArgsFactory(
        args_match=SimpleNamespace(
            number_of_games_player_one_white=1,
            number_of_games_player_one_black=2,
        ),
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
        args_game=SimpleNamespace(game_kind=GameKind.CHESS),
    )

    first_mapping, _, _ = factory.generate_game_args(0)
    second_mapping, _, _ = factory.generate_game_args(1)

    assert first_mapping[Color.WHITE].player_args.name == "player-one"
    assert first_mapping[Color.BLACK].player_args.name == "player-two"
    assert second_mapping[Color.WHITE].player_args.name == "player-two"
    assert second_mapping[Color.BLACK].player_args.name == "player-one"


def test_generate_game_args_merges_seed_and_tracks_match_completion() -> None:
    """Freeze the current per-game seed merge and completion counting behavior."""
    factory = GameArgsFactory(
        args_match=SimpleNamespace(
            number_of_games_player_one_white=1,
            number_of_games_player_one_black=2,
        ),
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
        args_game=SimpleNamespace(game_kind=GameKind.CHESS),
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
        args_match=SimpleNamespace(
            number_of_games_player_one_white=1,
            number_of_games_player_one_black=0,
        ),
        args_player_one=PlayerArgs(
            name="solo-player",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        args_player_two=None,
        seed_=5,
        args_game=SimpleNamespace(game_kind=GameKind.INTEGER_REDUCTION),
    )

    assignment, _, merged_seed = factory.generate_game_args(0)

    assert set(assignment) == {SOLO}
    assert assignment[SOLO].player_args.name == "solo-player"
    assert merged_seed == unique_int_from_list([5, 0])
    assert factory.is_match_finished() is True
