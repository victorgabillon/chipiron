"""Builders for script types and arguments from GUI choices."""

from typing import Any

from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from parsley import make_partial_dataclass_with_optional_paths

from chipiron import scripts
from chipiron.environments.checkers.starting_position_args import (
    CheckersStandardStartingPositionArgs,
)
from chipiron.environments.chess.starting_position_args import (
    FenStartingPositionArgs,
    StartingPositionArgsType,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.match.match_args import MatchArgs
from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs
from chipiron.games.domain.match.match_tag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

from .models import ArgsChosenByUser, ScriptGUIType
from .registries import starting_positions_for_game


def format_gui_args_for_display(gui_args: Any) -> str:
    """Format GUI arguments for logging."""
    if gui_args is None:
        return "None"

    args_parts: list[str] = []

    if hasattr(gui_args, "gui") and gui_args.gui:
        args_parts.append("GUI: Enabled")

    if hasattr(gui_args, "match_args") and gui_args.match_args:
        match_args = gui_args.match_args
        if hasattr(match_args, "player_one") and match_args.player_one:
            args_parts.append(
                f"Player One: {match_args.player_one.name if hasattr(match_args.player_one, 'name') else match_args.player_one}"
            )
        if hasattr(match_args, "player_two") and match_args.player_two:
            args_parts.append(
                f"Player Two: {match_args.player_two.name if hasattr(match_args.player_two, 'name') else match_args.player_two}"
            )
        if hasattr(match_args, "match_setting") and match_args.match_setting:
            args_parts.append(
                f"Match Setting: {match_args.match_setting.name if hasattr(match_args.match_setting, 'name') else match_args.match_setting}"
            )

    if hasattr(gui_args, "base_script_args") and gui_args.base_script_args:
        base_args = gui_args.base_script_args
        if hasattr(base_args, "profiling") and base_args.profiling is not None:
            args_parts.append(
                f"Profiling: {'Enabled' if base_args.profiling else 'Disabled'}"
            )
        if hasattr(base_args, "seed") and base_args.seed is not None:
            args_parts.append(f"Random Seed: {base_args.seed}")

    return "\n    ".join(args_parts) if args_parts else "Default settings"


def generate_inputs(
    args_chosen_by_user: ArgsChosenByUser,
) -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Generate script type and arguments from GUI selections."""
    partial_op_match_script_args = make_partial_dataclass_with_optional_paths(
        cls=MatchScriptArgs
    )
    partial_op_match_args = make_partial_dataclass_with_optional_paths(cls=MatchArgs)
    partial_op_match_settings_args = make_partial_dataclass_with_optional_paths(
        cls=MatchSettingsArgs
    )
    partial_op_game_args = make_partial_dataclass_with_optional_paths(cls=GameArgs)
    partial_op_player_args = make_partial_dataclass_with_optional_paths(cls=PlayerArgs)
    partial_op_base_script_args = make_partial_dataclass_with_optional_paths(
        cls=BaseScriptArgs
    )
    partial_op_tree_and_value_player_args = make_partial_dataclass_with_optional_paths(
        cls=TreeAndValuePlayerArgs
    )
    partial_op_tree_branch_limit_args = make_partial_dataclass_with_optional_paths(
        cls=TreeBranchLimitArgs
    )

    match args_chosen_by_user.type:
        case ScriptGUIType.PLAY_OR_WATCH_A_GAME:
            if args_chosen_by_user.game_kind == GameKind.CHESS:
                fen = starting_positions_for_game(args_chosen_by_user.game_kind)[
                    args_chosen_by_user.starting_position_key
                ]
                game_args = partial_op_game_args(
                    game_kind=args_chosen_by_user.game_kind,
                    starting_position=FenStartingPositionArgs(
                        type=StartingPositionArgsType.FEN,
                        fen=fen,
                    ),
                )
            else:
                game_args = partial_op_game_args(
                    game_kind=args_chosen_by_user.game_kind,
                    starting_position=CheckersStandardStartingPositionArgs(),
                )

            gui_args = partial_op_match_script_args(
                gui=True,
                base_script_args=partial_op_base_script_args(profiling=False, seed=0),
                match_args=partial_op_match_args(
                    match_setting=MatchConfigTag.DUDA,
                    match_setting_overwrite=partial_op_match_settings_args(
                        game_args=game_args
                    ),
                ),
            )
            config_file_name = "package://scripts/one_match/inputs/human_play_against_computer/exp_options.yaml"

            gui_args.match_args.player_one = PlayerConfigTag(
                args_chosen_by_user.player_type_white
            )
            gui_args.match_args.player_two = PlayerConfigTag(
                args_chosen_by_user.player_type_black
            )

            if args_chosen_by_user.game_kind == GameKind.CHESS:
                if (
                    args_chosen_by_user.player_type_white != PlayerConfigTag.GUI_HUMAN
                    and args_chosen_by_user.strength_white is not None
                ):
                    gui_args.match_args.player_one_overwrite = partial_op_player_args(
                        main_move_selector=partial_op_tree_and_value_player_args(
                            type=MoveSelectorTypes.TREE_AND_VALUE,
                            stopping_criterion=partial_op_tree_branch_limit_args(
                                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                                tree_branch_limit=4
                                * 10**args_chosen_by_user.strength_white,
                            ),
                        ),
                    )
                if (
                    args_chosen_by_user.player_type_black != PlayerConfigTag.GUI_HUMAN
                    and args_chosen_by_user.strength_black is not None
                ):
                    gui_args.match_args.player_two_overwrite = partial_op_player_args(
                        main_move_selector=partial_op_tree_and_value_player_args(
                            type=MoveSelectorTypes.TREE_AND_VALUE,
                            stopping_criterion=partial_op_tree_branch_limit_args(
                                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                                tree_branch_limit=4
                                * 10**args_chosen_by_user.strength_black,
                            ),
                        ),
                    )

            script_type = scripts.ScriptType.ONE_MATCH
        case ScriptGUIType.TREE_VISUALIZATION:
            config_file_name = "scripts/tree_visualization/inputs/base/exp_options.yaml"
            gui_args = None
            script_type = scripts.ScriptType.TREE_VISUALIZATION

    chipiron_logger.info(
        "GUI Configuration Selected:\n"
        "  Script Type: %s\n"
        "  Arguments:\n    %s\n"
        "  Config File: %s",
        script_type.name if hasattr(script_type, "name") else script_type,
        format_gui_args_for_display(gui_args),
        config_file_name,
    )

    return script_type, gui_args, config_file_name
