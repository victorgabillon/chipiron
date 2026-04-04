"""Characterization tests for current white/black factory and result assumptions."""

from __future__ import annotations

from types import SimpleNamespace

from test_support.import_compat import bootstrap_test_imports

bootstrap_test_imports()

from valanga import Color

from chipiron.games.domain.game.final_game_result import FinalGameResult
from chipiron.games.domain.game.game_args_factory import GameArgsFactory
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.players import PlayerArgs
from chipiron.utils.small_tools import unique_int_from_list


class EngineSelector:
    """Minimal non-human selector used by the factory characterization tests."""

    def is_human(self) -> bool:
        return False


def test_generate_game_args_assigns_white_and_black_from_player_one_quota() -> None:
    """Freeze the current white/black assignment logic for match scheduling."""
    factory = GameArgsFactory(
        args_match=SimpleNamespace(
            number_of_games_player_one_white=1,
            number_of_games_player_one_black=2,
        ),
        args_player_one=PlayerArgs(name="player-one", main_move_selector=EngineSelector()),
        args_player_two=PlayerArgs(name="player-two", main_move_selector=EngineSelector()),
        seed_=11,
        args_game=SimpleNamespace(kind="fixture-game"),
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
        args_player_one=PlayerArgs(name="player-one", main_move_selector=EngineSelector()),
        args_player_two=PlayerArgs(name="player-two", main_move_selector=EngineSelector()),
        seed_=17,
        args_game=SimpleNamespace(kind="fixture-game"),
    )

    _, _, first_seed = factory.generate_game_args(0)
    _, _, second_seed = factory.generate_game_args(1)
    assert first_seed == unique_int_from_list([17, 0])
    assert second_seed == unique_int_from_list([17, 1])
    assert factory.is_match_finished() is False

    factory.generate_game_args(2)
    assert factory.is_match_finished() is True


def test_match_results_counts_wins_by_player_identity_given_the_white_player() -> None:
    """Freeze the current white-player-based accounting in MatchResults."""
    results = MatchResults(
        player_one_name_id="player-one",
        player_two_name_id="player-two",
    )

    results.add_result_one_game(
        white_player_name_id="player-two",
        game_result=FinalGameResult.WIN_FOR_BLACK,
    )

    simple = results.get_simple_result()
    assert simple.player_one_wins == 1
    assert simple.player_two_wins == 0
    assert simple.draws == 0
