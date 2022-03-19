import sys
from src.displays.gui import MainWindow
from PyQt5.QtWidgets import *
from scripts.script import Script
from shutil import copyfile
from src.extra_tools.small_tools import mkdir, yaml_fetch_args_in_file,rec_merge_dic
from src.games.match_factories import MatchManagerFactory
from src.my_random import set_seeds
import multiprocessing
from src.players.boardevaluators.table_base.factory import create_syzygy_thread
import queue

class OneMatchScript(Script):
    """
    Script that plays a match between two players

    """
    default_param_dict = Script.default_param_dict | {'config_file_name': 'scripts/one_match/exp_options.yaml',
                                                      'seed': 0,
                                                      'gui': True,
                                                      'file_name_player_one': 'RecurZipfBase3.yaml',
                                                      'player_one': {},
                                                      'file_name_player_two': 'RecurZipfBase4.yaml',
                                                      'player_two': {},
                                                      'file_name_match_setting': 'setting_duda.yaml',
                                                      'match': {},
                                                      }

    base_experiment_output_folder = Script.base_experiment_output_folder + 'one_match/one_match_outputs/'

    def __init__(self, gui_args):
        """
        Builds the OneMatchScript object
        """
        # Getting % usage of virtual_memory ( 3rd field)
        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        super().__init__(gui_args)

        # taking care of random
        set_seeds(seed=self.args['seed'])

        file_name_player_one = self.args['file_name_player_one']
        file_name_player_two = self.args['file_name_player_two']
        if self.args['profiling']:
            self.args['max_half_move'] = 1
            file_name_match_setting = 'setting_jime.yaml'
        else:
            file_name_match_setting = self.args['file_name_match_setting']

        self.fetch_args_and_create_output_folder(file_name_player_one, file_name_player_two, file_name_match_setting)

        # Creation of the Syzygy table for perfect play in low pieces cases, needed by the GameManager
        # and can also be used by the players
        syzygy_mailbox = create_syzygy_thread()

        main_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()

        match_manager_factory = MatchManagerFactory(self.args['match'], self.args['player_one'],
                                                    self.args['player_two'], syzygy_mailbox,
                                                    self.experiment_output_folder, self.args['seed'],
                                                    main_thread_mailbox)

        if self.args['gui']:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox = multiprocessing.Manager().Queue()
            self.chess_gui = QApplication(sys.argv)
            self.window = MainWindow(gui_thread_mailbox, main_thread_mailbox)
            match_manager_factory.subscribe(gui_thread_mailbox)

        self.match_manager = match_manager_factory.create()

    def fetch_args_and_create_output_folder(self, file_name_player_one, file_name_player_two, file_name_match_setting):
        """
        From the names of the config file for players and match setting, open the config files, loads the arguments
         and copy the config files in the experiment folder.
        Args:
            file_name_player_one:
            file_name_player_two:
            file_name_match_setting:

        Returns:

        """
        path_player_one = 'data/players/' + file_name_player_one
        player_one_yaml = yaml_fetch_args_in_file(path_player_one)
        self.args['player_one'] = rec_merge_dic(player_one_yaml, self.args['player_one'])
        path_player_two = 'data/players/' + file_name_player_two
        player_two_yaml = yaml_fetch_args_in_file(path_player_two)
        self.args['player_two'] = rec_merge_dic(player_two_yaml, self.args['player_two'])
        path_match_setting = 'data/settings/OneMatch/' + file_name_match_setting
        match_setting_yaml = yaml_fetch_args_in_file(path_match_setting)
        self.args['match'] = rec_merge_dic(match_setting_yaml, self.args['match'])

        file_game = self.args['match']['game_setting_file']
        path_game_setting = 'data/settings/GameSettings/' + file_game

        path_games = self.experiment_output_folder + '/games'
        mkdir(path_games)
        copyfile(path_game_setting, self.experiment_output_folder + '/' + file_game)
        copyfile(path_player_one, self.experiment_output_folder + '/' + file_name_player_one)
        copyfile(path_player_two, self.experiment_output_folder + '/' + file_name_player_two)
        copyfile(path_match_setting, self.experiment_output_folder + '/' + file_name_match_setting)

    def run(self):
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
