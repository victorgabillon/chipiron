"""
Module to fetch, modify and convert the match settings and game settings.
"""

import os
from dataclasses import asdict, dataclass
from shutil import copyfile

import parsley_coco
import yaml

import chipiron as ch
import chipiron.games.game as game
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.MatchTag import MatchConfigTag
from chipiron.utils import path

from .match_settings_args import MatchSettingsArgs


def fetch_match_games_args_convert_and_save(
    match_args: MatchArgs,
    profiling: bool = False,
    testing: bool = False,
    experiment_output_folder: path | None = None,
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

    if profiling:
        path_match_setting: path = MatchConfigTag.Cubo.get_yaml_file_path()
        match_setting: MatchSettingsArgs = (
            parsley_coco.resolve_yaml_file_to_base_dataclass(
                yaml_path=path_match_setting,
                base_cls=MatchSettingsArgs,
            )
        )
        match_args.match_setting = match_setting

    assert isinstance(match_args.match_setting, MatchSettingsArgs)
    file_game: path = match_args.match_setting.game_setting_file
    path_game_setting: path = os.path.join("data/settings/GameSettings", file_game)

    if experiment_output_folder is not None:
        path_games: path = os.path.join(experiment_output_folder, "games")
        ch.tool.mkdir_if_not_existing(path_games)
        copyfile(
            src=path_game_setting, dst=os.path.join(experiment_output_folder, file_game)
        )
        with open(os.path.join(experiment_output_folder, "match_setting"), "w") as f:
            yaml.dump(asdict(match_args), f)

    game_file_path: path = os.path.join(
        "data/settings/GameSettings", match_args.match_setting.game_setting_file
    )
    args_game: game.GameArgs = parsley_coco.resolve_yaml_file_to_base_dataclass(
        yaml_path=game_file_path, base_cls=game.GameArgs, raise_error_with_nones=False
    )

    if profiling:
        args_game.max_half_moves = 1
    else:
        args_game.max_half_moves = None

    return match_args.match_setting, args_game
