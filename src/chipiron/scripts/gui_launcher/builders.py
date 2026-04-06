"""Builders for script types and arguments from GUI choices."""

from collections.abc import Sequence
from typing import Any, Callable, cast

from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from parsley import make_partial_dataclass_with_optional_paths  # type: ignore[reportMissingImports]

from chipiron import scripts
from chipiron.environments.checkers.starting_position_args import (
    CheckersStandardStartingPositionArgs,
)
from chipiron.environments.chess.starting_position_args import (
    FenStartingPositionArgs,
    StartingPositionArgsType,
)
from chipiron.environments.integer_reduction.starting_position_args import (
    IntegerReductionValueStartingPositionArgs,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.match.match_args import MatchArgs
from chipiron.games.domain.match.match_role_schedule import (
    SoloMatchSchedule,
    TwoRoleMatchSchedule,
)
from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs
from chipiron.games.domain.match.match_tag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

from .models import ArgsChosenByUser, ParticipantSelection, ScriptGUIType
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


def _schedule_for_participants(
    participants: Sequence[ParticipantSelection],
) -> SoloMatchSchedule | TwoRoleMatchSchedule:
    """Choose a neutral schedule from the configured participant topology."""
    if len(participants) == 1:
        return SoloMatchSchedule(number_of_games=1)
    if len(participants) == 2:
        return TwoRoleMatchSchedule(
            number_of_games_player_one_on_first_role=1,
            number_of_games_player_one_on_second_role=0,
        )
    raise ValueError(
        f"Launcher currently supports only 1 or 2 participants, got {len(participants)}."
    )


def _game_args_from_user_choices(
    args_chosen_by_user: ArgsChosenByUser,
    *,
    partial_op_game_args: Callable[..., GameArgs],
) -> GameArgs:
    """Build game args from launcher state."""
    starting_positions = starting_positions_for_game(args_chosen_by_user.game_kind)
    starting_position_value = starting_positions[args_chosen_by_user.starting_position_key]

    if args_chosen_by_user.game_kind is GameKind.CHESS:
        return partial_op_game_args(
            game_kind=args_chosen_by_user.game_kind,
            starting_position=FenStartingPositionArgs(
                type=StartingPositionArgsType.FEN,
                fen=starting_position_value,
            ),
        )

    return partial_op_game_args(
        game_kind=args_chosen_by_user.game_kind,
        starting_position=(
            CheckersStandardStartingPositionArgs()
            if args_chosen_by_user.game_kind is GameKind.CHECKERS
            else IntegerReductionValueStartingPositionArgs(
                value=int(starting_position_value)
            )
        ),
    )


def _player_overwrite_from_participant(
    *,
    game_kind: GameKind,
    participant: ParticipantSelection,
    partial_op_player_args: Callable[..., PlayerArgs],
    partial_op_tree_and_value_player_args: Callable[..., TreeAndValuePlayerArgs],
    partial_op_tree_branch_limit_args: Callable[..., TreeBranchLimitArgs],
) -> PlayerArgs | None:
    """Build an optional player overwrite from launcher strength controls."""
    if game_kind is not GameKind.CHESS:
        return None
    if participant.player_tag.is_human() or participant.strength is None:
        return None

    return partial_op_player_args(
        main_move_selector=partial_op_tree_and_value_player_args(
            type=MoveSelectorTypes.TREE_AND_VALUE,
            stopping_criterion=partial_op_tree_branch_limit_args(
                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                tree_branch_limit=4 * 10**participant.strength,
            ),
        ),
    )


def generate_inputs(
    args_chosen_by_user: ArgsChosenByUser,
) -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Generate script type and arguments from GUI selections."""
    partial_op_match_script_args = cast(
        Callable[..., MatchScriptArgs],
        make_partial_dataclass_with_optional_paths(cls=MatchScriptArgs),
    )
    partial_op_match_args = cast(
        Callable[..., MatchArgs],
        make_partial_dataclass_with_optional_paths(cls=MatchArgs),
    )
    partial_op_match_settings_args = cast(
        Callable[..., MatchSettingsArgs],
        make_partial_dataclass_with_optional_paths(cls=MatchSettingsArgs),
    )
    partial_op_game_args = cast(
        Callable[..., GameArgs],
        make_partial_dataclass_with_optional_paths(cls=GameArgs),
    )
    partial_op_player_args = cast(
        Callable[..., PlayerArgs],
        make_partial_dataclass_with_optional_paths(cls=PlayerArgs),
    )
    partial_op_base_script_args = cast(
        Callable[..., BaseScriptArgs],
        make_partial_dataclass_with_optional_paths(cls=BaseScriptArgs),
    )
    partial_op_tree_and_value_player_args = cast(
        Callable[..., TreeAndValuePlayerArgs],
        make_partial_dataclass_with_optional_paths(cls=TreeAndValuePlayerArgs),
    )
    partial_op_tree_branch_limit_args = cast(
        Callable[..., TreeBranchLimitArgs],
        make_partial_dataclass_with_optional_paths(cls=TreeBranchLimitArgs),
    )

    match args_chosen_by_user.type:
        case ScriptGUIType.PLAY_OR_WATCH_A_GAME:
            participants = args_chosen_by_user.participants
            if not participants:
                raise ValueError("Launcher state must contain at least one participant.")

            player_one = participants[0]
            player_two = participants[1] if len(participants) > 1 else None

            gui_args = partial_op_match_script_args(
                gui=True,
                base_script_args=partial_op_base_script_args(profiling=False, seed=0),
                match_args=partial_op_match_args(
                    match_setting=MatchConfigTag.DUDA,
                    match_setting_overwrite=partial_op_match_settings_args(
                        schedule=_schedule_for_participants(participants),
                        game_args=_game_args_from_user_choices(
                            args_chosen_by_user,
                            partial_op_game_args=partial_op_game_args,
                        ),
                    ),
                ),
            )
            config_file_name = "package://scripts/one_match/inputs/human_play_against_computer/exp_options.yaml"

            gui_args.match_args.player_one = player_one.player_tag
            gui_args.match_args.player_two = (
                player_two.player_tag if player_two is not None else None
            )

            player_one_overwrite = _player_overwrite_from_participant(
                game_kind=args_chosen_by_user.game_kind,
                participant=player_one,
                partial_op_player_args=partial_op_player_args,
                partial_op_tree_and_value_player_args=(
                    partial_op_tree_and_value_player_args
                ),
                partial_op_tree_branch_limit_args=partial_op_tree_branch_limit_args,
            )
            if player_one_overwrite is not None:
                setattr(
                    gui_args.match_args,
                    "player_one_overwrite",
                    player_one_overwrite,
                )

            if player_two is not None:
                player_two_overwrite = _player_overwrite_from_participant(
                    game_kind=args_chosen_by_user.game_kind,
                    participant=player_two,
                    partial_op_player_args=partial_op_player_args,
                    partial_op_tree_and_value_player_args=(
                        partial_op_tree_and_value_player_args
                    ),
                    partial_op_tree_branch_limit_args=partial_op_tree_branch_limit_args,
                )
                if player_two_overwrite is not None:
                    setattr(
                        gui_args.match_args,
                        "player_two_overwrite",
                        player_two_overwrite,
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
