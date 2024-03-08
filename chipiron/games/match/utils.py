from chipiron.utils import path
import chipiron as ch
import os
from shutil import copyfile
from .match_settings_args import MatchSettingsArgs
import chipiron.games.game as game
from chipiron.utils.small_tools import fetch_args_modify_and_convert


def fetch_match_games_args_convert_and_save(
        file_name_match_setting: path,
        profiling: bool = False,
        experiment_output_folder: path | None = None,
        modification: dict | None = None,
) -> tuple[MatchSettingsArgs, game.GameArgs]:
    file_name_match_setting: path

    if profiling:
        file_name_match_setting = 'setting_jime.yaml'
    else:
        file_name_match_setting = file_name_match_setting

    path_match_setting: str = os.path.join('data/settings/OneMatch', file_name_match_setting)
    match_args: MatchSettingsArgs = fetch_args_modify_and_convert(
        path_to_file=path_match_setting,
        modification=modification,
        dataclass_name=MatchSettingsArgs  # pycharm raises a warning here(hoping its beacause p
        # ycharm does not understand well annoation in 3.12 yet)
    )

    file_game: str = match_args.game_setting_file
    path_game_setting: str = os.path.join('data/settings/GameSettings', file_game)

    if experiment_output_folder is not None:
        path_games: str = os.path.join(experiment_output_folder, 'games')
        ch.tool.mkdir(path_games)
        copyfile(src=path_game_setting,
                 dst=os.path.join(experiment_output_folder, file_game))
        copyfile(src=path_match_setting,
                 dst=os.path.join(experiment_output_folder, file_name_match_setting))

    game_file_path: path = os.path.join('data/settings/GameSettings', match_args.game_setting_file)
    args_game: game.GameArgs = fetch_args_modify_and_convert(
        path_to_file=game_file_path,
        dataclass_name=game.GameArgs
    )

    if profiling:
        args_game.max_half_moves = 1
    else:
        args_game.max_half_moves = None

    return match_args, args_game
