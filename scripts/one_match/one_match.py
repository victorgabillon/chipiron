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
from chipiron.games.match_factories import create_match_manager, MatchArgs
from utils import path
from dataclasses import dataclass, field
import dacite
from players.factory import PlayerArgs


@dataclass
class OneMatchScriptArgs(ScriptArgs):
    """
    The input arguments needed by the one match script to run
    """
    # the seed
    seed: int = 0

    # whether to display the match in a GUI
    gui: bool = True

    # path to files with yaml config the players and the match setting.
    config_file_name: str | bytes | os.PathLike = 'scripts/one_match/inputs/base/exp_options.yaml'
    file_name_player_one: str | bytes | os.PathLike = 'RecurZipfBase3.yaml'
    file_name_player_two: str | bytes | os.PathLike = 'RecurZipfBase4.yaml'
    file_name_match_setting: str | bytes | os.PathLike = 'setting_duda.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict = field(default_factory=dict)
    player_two: dict = field(default_factory=dict)


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
        self.args: dict = self.base_script.initiate()
        self.one_match_args: OneMatchScriptArgs = dacite.from_dict(data_class=OneMatchScriptArgs,
                                                                   data=self.args)

        # taking care of random
        ch.set_seeds(seed=self.one_match_args.seed)

        if self.one_match_args.profiling:
            self.args['max_half_move'] = 1
            file_name_match_setting = 'setting_jime.yaml'
        else:
            file_name_match_setting = self.args['file_name_match_setting']

        self.fetch_args_and_create_output_folder(
            file_name_player_one=self.one_match_args.file_name_player_one,
            file_name_player_two=self.one_match_args.file_name_player_two,
            file_name_match_setting=file_name_match_setting
        )

        file_path: path = os.path.join('data/settings/GameSettings', self.args['match']['game_setting_file'])
        with open(file_path, 'r', encoding="utf-8") as file_game:
            args_game: dict = yaml.load(file_game, Loader=yaml.FullLoader)

        self.match_manager: ch.game.MatchManager = create_match_manager(
            args_match=self.args['match'],
            args_player_one=self.args['player_one'],
            args_player_two=self.args['player_two'],
            output_folder_path=self.args['experiment_output_folder'],
            seed=self.args['seed'],
            args_game=args_game,
            gui=self.args['gui']
        )

        if self.args['gui']:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()
            self.chess_gui: QApplication = QApplication(sys.argv)
            self.window: ch.disp.MainWindow = ch.disp.MainWindow(
                gui_mailbox=gui_thread_mailbox,
                main_thread_mailbox=self.match_manager.game_manager_factory.main_thread_mailbox
            )
            self.match_manager.subscribe(gui_thread_mailbox)

    def fetch_args_and_create_output_folder(self,
                                            file_name_player_one: str | bytes | os.PathLike,
                                            file_name_player_two: str | bytes | os.PathLike,
                                            file_name_match_setting: str | bytes | os.PathLike) -> None:
        """
        From the names of the config file for players and match setting, open the config files, loads the arguments
         and copy the config files in the experiment folder.
        Args:
            file_name_player_one:
            file_name_player_two:
            file_name_match_setting:

        Returns:

        """
        path_player_one: str = os.path.join('data/players/player_config', file_name_player_one)
        player_one_yaml: dict = ch.tool.yaml_fetch_args_in_file(path_player_one)
        merge_args_dict: dict = ch.tool.rec_merge_dic(player_one_yaml, self.one_match_args.player_one)

        # formatting the dictionary into the corresponding dataclass
        player_one_args: PlayerArgs = dacite.from_dict(data_class=PlayerArgs,
                                                       data=merge_args_dict)

        path_player_two: str = os.path.join('data/players/player_config', file_name_player_two)
        player_two_yaml: dict = ch.tool.yaml_fetch_args_in_file(path_player_two)
        self.args['player_two']: dict = ch.tool.rec_merge_dic(player_two_yaml, self.args['player_two'])

        path_match_setting: str = os.path.join('data/settings/OneMatch', file_name_match_setting)
        match_setting_yaml: dict = ch.tool.yaml_fetch_args_in_file(path_match_setting)
        self.args['match']: dict = ch.tool.rec_merge_dic(match_setting_yaml, self.args['match'])

        file_game: str = self.args['match']['game_setting_file']
        path_game_setting: str = 'data/settings/GameSettings/' + file_game

        path_games: str = self.args['experiment_output_folder'] + '/games'
        ch.tool.mkdir(path_games)
        copyfile(path_game_setting, self.args['experiment_output_folder'] + '/' + file_game)
        copyfile(path_player_one, self.args['experiment_output_folder'] + '/' + file_name_player_one)
        copyfile(path_player_two, self.args['experiment_output_folder'] + '/' + file_name_player_two)
        copyfile(path_match_setting, self.args['experiment_output_folder'] + '/' + file_name_match_setting)

    def run(self) -> None:
        """
        Runs the match either with a GUI or not
        Returns:

        """

        # Qt Application needs to be in the main Thread, so we need to distinguish between GUI and no GUI
        if self.args['gui']:  # case with GUI
            # Launching the Match Manager in a Thread
            process_match_manager = multiprocessing.Process(target=self.match_manager.play_one_match)
            process_match_manager.start()

            # Qt Application launched in the main thread
            self.window.show()
            self.chess_gui.exec_()
        else:  # No GUI
            self.match_manager.play_one_match()

        # TODO check the good closing of processes
