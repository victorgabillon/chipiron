import os
from shutil import copyfile
from chipiron.players.player_args import PlayerArgs
from chipiron.utils.small_tools import fetch_args_modify_and_convert
from chipiron.utils import path
from typing import Any


def fetch_player_args_convert_and_save(
        file_name_player: path,
        modification_player: dict[Any, Any] | None = None,
        experiment_output_folder: str | bytes | os.PathLike | None = None,
        from_data_folder: bool = True
) -> PlayerArgs:
    """
    From the names of the config file for players and match setting, open the config files, loads the arguments
     apply mofictions and copy the config files in the experiment folder.


    Returns:

    """
    path_player: str
    if from_data_folder:
        path_player = os.path.join('data/players/player_config', file_name_player)
    else:
        path_player = file_name_player

    player_args: PlayerArgs = fetch_args_modify_and_convert(
        path_to_file=path_player,
        modification=modification_player,
        dataclass_name=PlayerArgs  # pycharm raises a warning here(hoping its beacause p
        # ycharm does not understand well annoation in 3.12 yet)
    )

    if experiment_output_folder is not None:
        file_name_player = file_name_player.split('/')[-1]
        copyfile(
            src=path_player,
            dst=os.path.join(experiment_output_folder, file_name_player)
        )

    return player_args


def fetch_two_players_args_convert_and_save(
        file_name_player_one,
        file_name_player_two,
        modification_player_one,
        modification_player_two,
        experiment_output_folder
) -> tuple[PlayerArgs, PlayerArgs]:
    """
    From the names of the config file for players and match setting, open the config files, loads the arguments
     apply mofictions and copy the config files in the experiment folder.


    Returns:

    """

    player_one_args: PlayerArgs = fetch_player_args_convert_and_save(
        file_name_player=file_name_player_one,
        modification_player=modification_player_one,
        experiment_output_folder=experiment_output_folder
    )

    player_two_args: PlayerArgs = fetch_player_args_convert_and_save(
        file_name_player=file_name_player_two,
        modification_player=modification_player_two,
        experiment_output_folder=experiment_output_folder
    )

    return player_one_args, player_two_args
