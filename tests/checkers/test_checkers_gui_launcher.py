from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.gui_launcher.logic import apply_game_kind_defaults
from chipiron.scripts.gui_launcher.models import ArgsChosenByUser
from chipiron.scripts.gui_launcher.registries import player_options_for_game


def test_checkers_registry_includes_random_player_option() -> None:
    options = player_options_for_game(GameKind.CHECKERS)
    tags = {opt.tag for opt in options}
    assert PlayerConfigTag.GUI_HUMAN in tags
    assert PlayerConfigTag.RANDOM in tags


def test_args_defaults_remain_chess_friendly() -> None:
    args = ArgsChosenByUser()
    assert args.game_kind is GameKind.CHESS
    assert args.player_type_white is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.player_type_black is PlayerConfigTag.RECUR_ZIPF_BASE_3
    assert args.strength_white == 1
    assert args.strength_black == 1


def test_apply_game_kind_defaults_sets_checkers_human_defaults() -> None:
    args = ArgsChosenByUser()
    args.game_kind = GameKind.CHECKERS
    apply_game_kind_defaults(args)
    assert args.player_type_white is PlayerConfigTag.GUI_HUMAN
    assert args.player_type_black is PlayerConfigTag.GUI_HUMAN
    assert args.strength_white is None
    assert args.strength_black is None


def test_apply_game_kind_defaults_resets_chess_defaults() -> None:
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
