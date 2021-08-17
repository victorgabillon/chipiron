import yaml
import os
from datetime import datetime
from shutil import copyfile
import sys
from src.games.match_manager import MatchManager
from src.players.player import Player
from src.players.boardevaluators.syzygy import Syzygy
from src.displays.gui import MainWindow
import global_variables
from PyQt5.QtWidgets import *
from scripts.script import Script
from src.games.game_manager import GameManager
from src.chessenvironment.boards.board import BoardChi


class OneMatchScript(Script):

    def __init__(self):
        super().__init__()

    def run(self):
        file_name_player_one = 'best_0.yaml'
        file_name_player_one = 'RecurZipfBase.yaml'
        file_name_player_two = 'RecurZipfBase.yaml'
        if global_variables.profiling_bool:
            file_name_match_setting = 'setting_jime.yaml'
        else:
            file_name_match_setting = 'setting_duda.yaml'
        path_player_one = 'chipiron/runs/players/best_players/' + file_name_player_one
        path_player_one = 'chipiron/runs/players/' + file_name_player_one
        path_player_two = 'chipiron/runs/players/' + file_name_player_two
        path_match_setting = 'chipiron/runs/OneMatch/' + file_name_match_setting

        with open(path_match_setting, 'r') as fileMatch:
            args_match = yaml.load(fileMatch, Loader=yaml.FullLoader)
            print(args_match)

        with open(path_player_one, 'r') as filePlayerOne:
            args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player_one)

        with open(path_player_two, 'r') as filePlayerTwo:
            args_player_two = yaml.load(filePlayerTwo, Loader=yaml.FullLoader)
            print(args_player_two)

        fileGame = args_match['game_setting_file']
        path_game_setting = 'chipiron/runs/GameSettings/' + fileGame

        now = datetime.now()  # current date and time
        path_directory = "chipiron/runs/runsOutput/" + now.strftime("%A-%m-%d-%Y--%H:%M:%S:%f")
        path_games = path_directory + '/games'
        try:
            os.mkdir(path_directory)
            os.mkdir(path_games)
        except OSError:
            print("Creation of the directory %s failed" % path_directory)
        else:
            print("Successfully created the directory %s " % path_directory)
        copyfile(path_game_setting, path_directory + '/' + fileGame)
        copyfile(path_player_one, path_directory + '/' + file_name_player_one)
        copyfile(path_player_two, path_directory + '/' + file_name_player_two)
        copyfile(path_match_setting, path_directory + '/' + file_name_match_setting)


        syzygy = Syzygy('')
        player_one = Player(args_player_one, syzygy)
        player_two = Player(args_player_two, syzygy)
        board = BoardChi()
        game_manager = GameManager(board, syzygy, path_to_store_result=path_games + '/')
        match_manager = MatchManager(args_match, player_one, player_two, game_manager, path_directory)
        print('deterministic_behavior', global_variables.deterministic_behavior)

        if global_variables.profiling_bool:
            match_manager.play_one_match()

        if not global_variables.profiling_bool:
            chessGui = QApplication(sys.argv)
            window = MainWindow(match_manager)
            window.show()
            chessGui.exec_()
