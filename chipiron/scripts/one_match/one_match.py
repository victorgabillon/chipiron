import sys
from src.players.boardevaluators.syzygy import SyzygyTable
from src.displays.gui import MainWindow
from PyQt5.QtWidgets import *
from scripts.script import Script
from shutil import copyfile
from src.extra_tools.small_tools import mkdir, yaml_fetch_args_in_file
from src.games.match_factories import create_match_manager


class OneMatchScript(Script):
    """
    Script that plays a match between two players

    """
    default_param_dict = Script.default_param_dict | \
                         {'config_file_name': 'chipiron/scripts/one_match/exp_options.yaml',
                          'deterministic_behavior': False,
                          'deterministic_mode': 'SEED_FIXED_EVERY_MOVE',
                          'seed_fixing_type': 'FIX_SEED_WITH_CONSTANT',
                          'seed': 10,
                          'file_name_player_one': 'RecurZipfBase3.yaml',
                          'file_name_player_two': 'RecurZipfBase3.yaml',
                          'file_name_match_setting': 'setting_duda.yaml'}
    base_experiment_output_folder = Script.base_experiment_output_folder + 'match_outputs/'

    def __init__(self):
        super().__init__()
        file_name_player_one = self.args['file_name_player_one']
        file_name_player_two = self.args['file_name_player_two']
        if self.args['profiling']:
            self.args['max_half_move'] = 1
            file_name_match_setting = 'setting_jime.yaml'
        else:
            file_name_match_setting = self.args['file_name_match_setting']

        args_match, args_player_one, args_player_two = self.fetch_args_and_create_output_folder(
            file_name_player_one, file_name_player_two, file_name_match_setting)

        syzygy_table = SyzygyTable('')
        self.match_manager = create_match_manager(args_match, args_player_one, args_player_two, syzygy_table,
                                                  self.experiment_output_folder)

    def fetch_args_and_create_output_folder(self, file_name_player_one, file_name_player_two, file_name_match_setting):
        path_player_one = 'chipiron/data/players/' + file_name_player_one
        args_player_one = yaml_fetch_args_in_file(path_player_one)
        path_player_two = 'chipiron/data/players/' + file_name_player_two
        args_player_two = yaml_fetch_args_in_file(path_player_two)
        path_match_setting = 'chipiron/data/settings/OneMatch/' + file_name_match_setting
        args_match = yaml_fetch_args_in_file(path_match_setting)

        file_game = args_match['game_setting_file']
        path_game_setting = 'chipiron/data/settings/GameSettings/' + file_game

        path_games = self.experiment_output_folder + '/games'
        mkdir(path_games)
        copyfile(path_game_setting, self.experiment_output_folder + '/' + file_game)
        copyfile(path_player_one, self.experiment_output_folder + '/' + file_name_player_one)
        copyfile(path_player_two, self.experiment_output_folder + '/' + file_name_player_two)
        copyfile(path_match_setting, self.experiment_output_folder + '/' + file_name_match_setting)
        return args_match, args_player_one, args_player_two

    def run(self):
        if self.args['profiling']:
            self.match_manager.play_one_match()
        else:
            chess_gui = QApplication(sys.argv)
            window = MainWindow(self.match_manager)
            window.show()
            chess_gui.exec_()
