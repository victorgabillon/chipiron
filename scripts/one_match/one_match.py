"""
the one match script
"""
import sys
import queue
from shutil import copyfile
import os
import multiprocessing
import yaml
from PyQt5.QtWidgets import QApplication
from scripts.script import Script, ScriptArgs
import chipiron as ch
from chipiron.games.match_factories import create_match_manager, MatchArgs, GameArgs
from utils import path
from dataclasses import dataclass, field
import dacite
from chipiron.players.factory import PlayerArgs
from utils.is_dataclass import IsDataclass, DataClass
from typing import Type, LiteralString


@dataclass
class OneMatchScriptArgs(ScriptArgs):
    """
    The input arguments needed by the one match script to run
    """
    # the seed
    seed: int = 0

    # whether to display the match in a GUI
    gui: bool = True

    experiment_output_folder: str | bytes | os.PathLike = None

    # path to files with yaml config the players and the match setting.
    config_file_name: str | bytes | os.PathLike = 'scripts/one_match/inputs/base/exp_options.yaml'
    file_name_player_one: str | bytes | os.PathLike = 'RecurZipfBase3.yaml'
    file_name_player_two: str | bytes | os.PathLike = 'RecurZipfBase4.yaml'
    file_name_match_setting: str | bytes | os.PathLike = 'setting_duda.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict = field(default_factory=dict)
    player_two: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)


def fetch_args_modify_and_convert[ADataclass: IsDataclass](
        path_to_file: str | bytes | os.PathLike,
        modification: dict,
        dataclass_name: Type[ADataclass]
) -> ADataclass:
    file_args: dict = ch.tool.yaml_fetch_args_in_file(path_to_file)
    merged_args_dict: dict = ch.tool.rec_merge_dic(file_args, modification)

    # formatting the dictionary into the corresponding dataclass
    dataclass_args: ADataclass = dacite.from_dict(data_class=dataclass_name,
                                                  data=merged_args_dict)

    return dataclass_args


class OneMatchScript:
    """
    Script that plays a match between two players

    """

    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder, 'one_match/outputs/')
    base_script: Script

    def __init__(
            self,
            base_script: Script,
    ) -> None:
        """
        Builds the OneMatchScript object
        """

        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        args_dict: dict = self.base_script.initiate()

        # Converting the args in the standardized dataclass
        self.args: OneMatchScriptArgs = dacite.from_dict(data_class=OneMatchScriptArgs,
                                                         data=args_dict)

        # Recovering args from yaml file for player and merging with extra args and converting to standardized dataclass
        player_one_args: PlayerArgs
        player_two_args: PlayerArgs
        player_one_args, player_two_args = self.fetch_player_args_convert_and_save()

        # Recovering args from yaml file for match and game and merging with extra args and converting
        # to standardized dataclass
        match_args: MatchArgs
        game_args: GameArgs
        match_args, game_args = self.fetch_match_games_args_convert_and_save()

        # taking care of random
        ch.set_seeds(seed=self.args.seed)

        self.match_manager: ch.game.MatchManager = create_match_manager(
            args_match=match_args,
            args_player_one=player_one_args,
            args_player_two=player_two_args,
            output_folder_path=self.args.experiment_output_folder,
            seed=self.args.seed,
            args_game=game_args,
            gui=self.args.gui
        )

        if self.args.gui:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()
            self.chess_gui: QApplication = QApplication(sys.argv)
            self.window: ch.disp.MainWindow = ch.disp.MainWindow(
                gui_mailbox=gui_thread_mailbox,
                main_thread_mailbox=self.match_manager.game_manager_factory.main_thread_mailbox
            )
            self.match_manager.subscribe(gui_thread_mailbox)

    def fetch_player_args_convert_and_save(self) -> tuple[PlayerArgs, PlayerArgs]:
        """
        From the names of the config file for players and match setting, open the config files, loads the arguments
         apply mofictions and copy the config files in the experiment folder.


        Returns:

        """

        path_player_one: str = os.path.join('data/players/player_config', self.args.file_name_player_one)
        path_player_two: str = os.path.join('data/players/player_config', self.args.file_name_player_two)

        player_one_args: PlayerArgs = fetch_args_modify_and_convert(
            path_to_file=path_player_one,
            modification=self.args.player_one,
            dataclass_name=PlayerArgs  # pycharm raises a warning here(hoping its beacause p
            # ycharm does not understand well annoation in 3.12 yet)
        )

        #  player_one_yaml: dict = ch.tool.yaml_fetch_args_in_file(path_player_one)
        # merge_args_dict_one: dict = ch.tool.rec_merge_dic(player_one_yaml, self.one_match_args.player_one)

        #  # formatting the dictionary into the corresponding dataclass
        #  player_one_args: PlayerArgs = dacite.from_dict(data_class=PlayerArgs,
        #                                                data=merge_args_dict_one)

        player_two_args: PlayerArgs = fetch_args_modify_and_convert(
            path_to_file=path_player_two,
            modification=self.args.player_two,
            dataclass_name=PlayerArgs  # pycharm raises a warning here(hoping its beacause p
            # ycharm does not understand well annoation in 3.12 yet)
        )

        copyfile(src=path_player_one,
                 dst=os.path.join(self.args.experiment_output_folder, self.args.file_name_player_one))
        copyfile(src=path_player_two,
                 dst=os.path.join(self.args.experiment_output_folder, self.args.file_name_player_two))

        return player_one_args, player_two_args

    def fetch_match_games_args_convert_and_save(self) -> tuple[MatchArgs, GameArgs]:
        file_name_match_setting: str | bytes | os.PathLike
        max_half_move: int | None
        if self.args.profiling:
            max_half_move = 1
            file_name_match_setting = 'setting_jime.yaml'
        else:
            max_half_move = None
            file_name_match_setting = self.args.file_name_match_setting

        path_match_setting: str = os.path.join('data/settings/OneMatch', file_name_match_setting)
        match_args: MatchArgs = fetch_args_modify_and_convert(
            path_to_file=path_match_setting,
            modification=self.args.match,
            dataclass_name=MatchArgs  # pycharm raises a warning here(hoping its beacause p
            # ycharm does not understand well annoation in 3.12 yet)
        )

        file_game: str = match_args.game_setting_file
        path_game_setting: str = 'data/settings/GameSettings/' + file_game

        path_games: str = self.args.experiment_output_folder + '/games'
        ch.tool.mkdir(path_games)
        copyfile(src=path_game_setting,
                 dst=os.path.join(self.args.experiment_output_folder, file_game))
        copyfile(src=path_match_setting,
                 dst=os.path.join(self.args.experiment_output_folder, file_name_match_setting))

        file_path: path = os.path.join('data/settings/GameSettings', match_args.game_setting_file)
        with open(file_path, 'r', encoding="utf-8") as file_game:
            args_game: dict = yaml.load(file_game, Loader=yaml.FullLoader)

        return match_args, args_game

    def run(self) -> None:
        """
        Runs the match either with a GUI or not
        Returns:

        """

        # Qt Application needs to be in the main Thread, so we need to distinguish between GUI and no GUI
        if self.args.gui:  # case with GUI
            # Launching the Match Manager in a Thread
            process_match_manager = multiprocessing.Process(target=self.match_manager.play_one_match)
            process_match_manager.start()

            # Qt Application launched in the main thread
            self.window.show()
            self.chess_gui.exec_()
        else:  # No GUI
            self.match_manager.play_one_match()

        # TODO check the good closing of processes
