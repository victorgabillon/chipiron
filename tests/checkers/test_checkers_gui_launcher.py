"""Focused launcher coverage for participant-driven GUI state."""

from typing import TYPE_CHECKING, Any, cast

import yaml

from chipiron.environments.types import GameKind
from chipiron.games.domain.match.match_role_schedule import (
    SoloMatchSchedule,
    TwoRoleMatchSchedule,
)
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.gui_launcher.builders import generate_inputs
from chipiron.scripts.gui_launcher.logic import apply_game_kind_defaults
from chipiron.scripts.gui_launcher.models import (
    ArgsChosenByUser,
    ParticipantSelection,
    ScriptGUIType,
)
from chipiron.scripts.gui_launcher.registries import (
    launcher_spec_for_game,
    player_options_for_game,
    starting_positions_for_game,
)
from chipiron.scripts.gui_launcher.ui_ctk import participant_row_models_for_state
from chipiron.utils.small_tools import resolve_package_path

if TYPE_CHECKING:
    from chipiron.scripts.one_match.one_match import MatchScriptArgs


def test_launcher_specs_encode_participant_topology() -> None:
    """Launcher specs should expose the per-game participant topology."""
    assert launcher_spec_for_game(GameKind.CHESS).participant_count == 2
    assert launcher_spec_for_game(GameKind.CHECKERS).participant_count == 2
    assert launcher_spec_for_game(GameKind.INTEGER_REDUCTION).participant_count == 1
    assert launcher_spec_for_game(GameKind.MORPION).participant_count == 1


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
    assert args.starting_position_key == "Standard"
    assert len(args.participants) == 2
    assert args.participants == list(
        launcher_spec_for_game(GameKind.CHESS).default_participants
    )


def test_apply_game_kind_defaults_sets_checkers_two_participant_defaults() -> None:
    """Checkers defaults should prefer two local human participants."""
    args = ArgsChosenByUser()
    args.game_kind = GameKind.CHECKERS
    apply_game_kind_defaults(args)
    assert len(args.participants) == 2
    assert args.participants == [
        ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
    ]


def test_integer_reduction_registry_and_defaults_support_solo_play() -> None:
    """Integer reduction should expose a solo launcher state."""
    options = player_options_for_game(GameKind.INTEGER_REDUCTION)
    tags = {opt.tag for opt in options}
    labels = {opt.label for opt in options}
    args = ArgsChosenByUser(game_kind=GameKind.INTEGER_REDUCTION)

    apply_game_kind_defaults(args)

    assert PlayerConfigTag.GUI_HUMAN in tags
    assert PlayerConfigTag.RANDOM in tags
    assert PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC in tags
    assert PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC_DEBUG in tags
    assert "Tree (basic eval + debug)" in labels
    assert starting_positions_for_game(GameKind.INTEGER_REDUCTION) == {
        "Small": "7",
        "Standard": "15",
        "Large": "31",
    }
    assert len(args.participants) == 1
    assert args.participants[0] == ParticipantSelection(
        player_tag=PlayerConfigTag.GUI_HUMAN,
    )


def test_morpion_registry_and_defaults_support_solo_play() -> None:
    """Morpion should expose a solo launcher state."""
    options = player_options_for_game(GameKind.MORPION)
    tags = {opt.tag for opt in options}
    args = ArgsChosenByUser(game_kind=GameKind.MORPION)

    apply_game_kind_defaults(args)

    assert PlayerConfigTag.GUI_HUMAN in tags
    assert PlayerConfigTag.RANDOM in tags
    assert PlayerConfigTag.MORPION_TREE_BASIC in tags
    assert starting_positions_for_game(GameKind.MORPION) == {"Standard": "5T"}
    assert len(args.participants) == 1
    assert args.participants[0] == ParticipantSelection(
        player_tag=PlayerConfigTag.GUI_HUMAN,
    )


def test_generate_inputs_configures_integer_reduction_as_single_game_match() -> None:
    """Launcher builder should set one solo game and leave player two unused."""
    _, gui_args, config_file_name = generate_inputs(
        ArgsChosenByUser(
            type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
            game_kind=GameKind.INTEGER_REDUCTION,
            participants=[
                ParticipantSelection(
                    player_tag=PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC
                ),
            ],
            starting_position_key="Standard",
        )
    )

    assert gui_args is not None
    typed_gui_args = cast("MatchScriptArgs", gui_args)
    match_args = cast("Any", typed_gui_args.match_args)
    assert match_args.player_one is PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC
    assert match_args.player_two is None
    assert (
        config_file_name
        == "package://scripts/one_match/inputs/gui_launcher/exp_options.yaml"
    )
    assert isinstance(
        match_args.match_setting_overwrite.schedule,
        SoloMatchSchedule,
    )
    assert match_args.match_setting_overwrite.schedule.number_of_games == 1


def test_generate_inputs_configures_morpion_as_single_game_match() -> None:
    """Launcher builder should set one solo Morpion game and leave player two unused."""
    _, gui_args, _ = generate_inputs(
        ArgsChosenByUser(
            type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
            game_kind=GameKind.MORPION,
            participants=[
                ParticipantSelection(player_tag=PlayerConfigTag.MORPION_TREE_BASIC),
            ],
            starting_position_key="Standard",
        )
    )

    assert gui_args is not None
    typed_gui_args = cast("MatchScriptArgs", gui_args)
    match_args = cast("Any", typed_gui_args.match_args)
    assert match_args.player_one is PlayerConfigTag.MORPION_TREE_BASIC
    assert match_args.player_two is None
    assert isinstance(
        match_args.match_setting_overwrite.schedule,
        SoloMatchSchedule,
    )
    assert match_args.match_setting_overwrite.schedule.number_of_games == 1


def test_generate_inputs_configures_checkers_with_two_role_schedule() -> None:
    """Launcher builder should construct an explicit 2-role schedule for checkers."""
    _, gui_args, _ = generate_inputs(
        ArgsChosenByUser(
            type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
            game_kind=GameKind.CHECKERS,
            participants=[
                ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
                ParticipantSelection(player_tag=PlayerConfigTag.RANDOM),
            ],
            starting_position_key="Standard",
        )
    )

    assert gui_args is not None
    typed_gui_args = cast("MatchScriptArgs", gui_args)
    match_args = cast("Any", typed_gui_args.match_args)
    assert match_args.player_one is PlayerConfigTag.GUI_HUMAN
    assert match_args.player_two is PlayerConfigTag.RANDOM
    assert isinstance(
        match_args.match_setting_overwrite.schedule,
        TwoRoleMatchSchedule,
    )
    assert (
        match_args.match_setting_overwrite.schedule.number_of_games_player_one_on_first_role
        == 1
    )
    assert (
        match_args.match_setting_overwrite.schedule.number_of_games_player_one_on_second_role
        == 0
    )


def test_launcher_uses_neutral_one_match_base_config() -> None:
    """Launcher base config should not seed player choices through legacy fields."""
    config_path = resolve_package_path(
        "package://scripts/one_match/inputs/gui_launcher/exp_options.yaml"
    )

    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["gui"] is True
    assert "match_args" not in config


def test_apply_game_kind_defaults_resets_chess_defaults() -> None:
    """Switching back to chess should restore chess launcher defaults."""
    args = ArgsChosenByUser(
        game_kind=GameKind.CHESS,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ],
    )
    apply_game_kind_defaults(args)
    assert args.participants == list(
        launcher_spec_for_game(GameKind.CHESS).default_participants
    )


def test_participant_row_models_follow_game_topology() -> None:
    """Pure UI row helpers should expose one row for solo and two for two-role games."""
    integer_reduction_rows = participant_row_models_for_state(
        ArgsChosenByUser(game_kind=GameKind.INTEGER_REDUCTION)
    )
    morpion_rows = participant_row_models_for_state(
        ArgsChosenByUser(game_kind=GameKind.MORPION)
    )
    chess_rows = participant_row_models_for_state(
        ArgsChosenByUser(game_kind=GameKind.CHESS)
    )
    checkers_rows = participant_row_models_for_state(
        ArgsChosenByUser(game_kind=GameKind.CHECKERS)
    )

    assert len(integer_reduction_rows) == 1
    assert integer_reduction_rows[0].label_text == "Solo"
    assert len(morpion_rows) == 1
    assert morpion_rows[0].label_text == "Solo"
    assert len(chess_rows) == 2
    assert len(checkers_rows) == 2
