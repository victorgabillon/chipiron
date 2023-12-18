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
from scripts.script import Script
import chipiron as ch
from chipiron.games.match_factories import create_match_manager
from utils import path


class OneMatchScript:
    """
    Script that plays a match between two players

    """
    default_param_dict = {
        'config_file_name': 'scripts/one_match/inputs/base/exp_options.yaml',
        'seed': 0,
        'gui': True,
        'file_name_player_one': 'RecurZipfBase3.yaml',
        'player_one': {},
        'file_name_player_two': 'RecurZipfBase4.yaml',
        'player_two': {},
        'file_name_match_setting': 'setting_duda.yaml',
        'match': {},
    }

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
        self.args: dict = self.base_script.initiate(default_param_dict=self.default_param_dict)

        # taking care of random
        ch.set_seeds(seed=self.args['seed'])

        file_name_player_one: str = self.args['file_name_player_one']
        file_name_player_two: str = self.args['file_name_player_two']
        file_name_match_setting: str
        if self.args['profiling']:
            self.args['max_half_move'] = 1
            file_name_match_setting = 'setting_jime.yaml'
        else:
            file_name_match_setting = self.args['file_name_match_setting']

        self.fetch_args_and_create_output_folder(
            file_name_player_one=file_name_player_one,
            file_name_player_two=file_name_player_two,
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
                                            file_name_player_one: str,
                                            file_name_player_two: str,
                                            file_name_match_setting: str) -> None:
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
        self.args['player_one']: dict = ch.tool.rec_merge_dic(player_one_yaml, self.args['player_one'])

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
        copyfile(path_player_two, self.args['experiment_output_folder']+ '/' + file_name_player_two)
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
