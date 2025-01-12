"""
Module to fetch, modify and convert the match settings and game settings.
"""

import os
from shutil import copyfile
from typing import Any

import chipiron as ch
import chipiron.games.game as game
from chipiron.utils import path
from chipiron.utils.small_tools import fetch_args_modify_and_convert

from .match_settings_args import MatchSettingsArgs


def fetch_match_games_args_convert_and_save(
    file_name_match_setting: path,
    profiling: bool = False,
    testing: bool = False,
    experiment_output_folder: path | None = None,
    modification: dict[Any, Any] | None = None,
) -> tuple[MatchSettingsArgs, game.GameArgs]:
    """
    Fetches, modifies, and converts the match settings and game settings.

    Args:
        file_name_match_setting (path): The path to the match settings file.
        profiling (bool, optional): Flag indicating whether profiling is enabled. Defaults to False.
        testing (bool, optional): Flag indicating whether testing is enabled. Defaults to False.
        experiment_output_folder (path | None, optional): The path to the experiment output folder. Defaults to None.
        modification (dict[Any, Any] | None, optional): Dictionary of modifications to apply. Defaults to None.

    Returns:
        tuple[MatchSettingsArgs, game.GameArgs]: A tuple containing the match settings and game settings.
    """
    file_name_match_setting_: path

    if profiling:
        file_name_match_setting_ = "setting_cubo.yaml"
        if testing:
            file_name_match_setting_ = "setting_cubo.yaml"
    else:
        file_name_match_setting_ = file_name_match_setting

    path_match_setting: str = os.path.join(
        "data/settings/OneMatch", file_name_match_setting_
    )
    match_args: MatchSettingsArgs = fetch_args_modify_and_convert(
        path_to_file=path_match_setting,
        modification=modification,
        dataclass_name=MatchSettingsArgs,
    )

    file_game: path = match_args.game_setting_file
    path_game_setting: path = os.path.join("data/settings/GameSettings", file_game)

    if experiment_output_folder is not None:
        path_games: path = os.path.join(experiment_output_folder, "games")
        ch.tool.mkdir(path_games)
        copyfile(
            src=path_game_setting, dst=os.path.join(experiment_output_folder, file_game)
        )
        copyfile(
            src=path_match_setting,
            dst=os.path.join(experiment_output_folder, file_name_match_setting_),
        )

    game_file_path: path = os.path.join(
        "data/settings/GameSettings", match_args.game_setting_file
    )
    args_game: game.GameArgs = fetch_args_modify_and_convert(
        path_to_file=game_file_path, dataclass_name=game.GameArgs
    )

    if profiling:
        args_game.max_half_moves = 1
    else:
        args_game.max_half_moves = None

    return match_args, args_game
