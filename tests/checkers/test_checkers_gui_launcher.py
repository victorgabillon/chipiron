"""Focused launcher coverage for checkers and integer reduction."""

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.gui_launcher.builders import generate_inputs
from chipiron.scripts.gui_launcher.logic import apply_game_kind_defaults
from chipiron.scripts.gui_launcher.models import ArgsChosenByUser, ScriptGUIType
from chipiron.scripts.gui_launcher.registries import (
    player_options_for_game,
    starting_positions_for_game,
)


def test_checkers_registry_includes_random_player_option() -> None:
    """Checkers registry should keep its current player options."""
    options = player_options_for_game(GameKind.CHECKERS)
    tags = {opt.tag for opt in options}
    assert PlayerConfigTag.GUI_HUMAN in tags
    assert PlayerConfigTag.RANDOM in tags
    assert PlayerConfigTag.CHECKERS_TREE_PIECECOUNT in tags


def test_args_defaults_remain_chess_friendly() -> None:
    """The default launcher args should stay chess-oriented."""
    args = ArgsChosenByUser()
    assert args.game_kind is GameKind.CHESS
    assert args.player_type_white is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.player_type_black is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.strength_white == 1
    assert args.strength_black == 1


def test_apply_game_kind_defaults_sets_checkers_human_defaults() -> None:
    """Checkers defaults should prefer two local human participants."""
    args = ArgsChosenByUser()
    args.game_kind = GameKind.CHECKERS
    apply_game_kind_defaults(args)
    assert args.player_type_white is PlayerConfigTag.GUI_HUMAN
    assert args.player_type_black is PlayerConfigTag.GUI_HUMAN
    assert args.strength_white is None
    assert args.strength_black is None


def test_integer_reduction_registry_and_defaults_support_solo_play() -> None:
    """Integer reduction should expose human/random options and human defaults."""
    options = player_options_for_game(GameKind.INTEGER_REDUCTION)
    tags = {opt.tag for opt in options}
    args = ArgsChosenByUser(game_kind=GameKind.INTEGER_REDUCTION)

    apply_game_kind_defaults(args)

    assert PlayerConfigTag.GUI_HUMAN in tags
    assert PlayerConfigTag.RANDOM in tags
    assert starting_positions_for_game(GameKind.INTEGER_REDUCTION) == {
        "Small": "7",
        "Standard": "15",
        "Large": "31",
    }
    assert args.player_type_white is PlayerConfigTag.GUI_HUMAN
    assert args.player_type_black is PlayerConfigTag.GUI_HUMAN
    assert args.strength_white is None
    assert args.strength_black is None


def test_generate_inputs_configures_integer_reduction_as_single_game_match() -> None:
    """Launcher builder should set one solo game and leave player two unused."""
    _, gui_args, _ = generate_inputs(
        ArgsChosenByUser(
            type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
            game_kind=GameKind.INTEGER_REDUCTION,
            player_type_white=PlayerConfigTag.RANDOM,
            player_type_black=PlayerConfigTag.GUI_HUMAN,
            starting_position_key="Standard",
        )
    )

    assert gui_args is not None
    assert gui_args.match_args.player_one is PlayerConfigTag.RANDOM
    assert gui_args.match_args.player_two is None
    assert gui_args.match_args.match_setting_overwrite.number_of_games_player_one_white == 1
    assert gui_args.match_args.match_setting_overwrite.number_of_games_player_one_black == 0


def test_apply_game_kind_defaults_resets_chess_defaults() -> None:
    """Switching back to chess should restore the old chess defaults."""
    args = ArgsChosenByUser(
        game_kind=GameKind.CHESS,
        player_type_white=PlayerConfigTag.GUI_HUMAN,
        strength_white=None,
        player_type_black=PlayerConfigTag.GUI_HUMAN,
        strength_black=None,
    )
    apply_game_kind_defaults(args)
    assert args.player_type_white is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.player_type_black is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.strength_white == 1
    assert args.strength_black == 1
