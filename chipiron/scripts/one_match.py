import yaml
import os
from datetime import datetime
from shutil import copyfile
import sys
from src.games.play_one_match import PlayOneMatch
from src.players.create_player import create_player
from src.chessenvironment.chess_environment import ChessEnvironment
from src.players.boardevaluators.syzygy import Syzygy
from src.displays.gui import MainWindow
import global_variables
from PyQt5.QtWidgets import *
from scripts.script import Script


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
            file_name_match_setting = 'setting_giri.yaml'
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
        pathDirectory = "chipiron/runs/runsOutput/" + now.strftime("%A-%m-%d-%Y--%H:%M:%S:%f")
        path_games = pathDirectory + '/games'
        try:
            os.mkdir(pathDirectory)
            os.mkdir(path_games)
        except OSError:
            print("Creation of the directory %s failed" % pathDirectory)
        else:
            print("Successfully created the directory %s " % pathDirectory)
        copyfile(path_game_setting, pathDirectory + '/' + fileGame)
        copyfile(path_player_one, pathDirectory + '/' + file_name_player_one)
        copyfile(path_player_two, pathDirectory + '/' + file_name_player_two)
        copyfile(path_match_setting, pathDirectory + '/' + file_name_match_setting)

        chess_simulator = ChessEnvironment()
        syzygy = Syzygy(chess_simulator, '')

        player_one = create_player(args_player_one, chess_simulator, syzygy)
        player_two = create_player(args_player_two, chess_simulator, syzygy)

        play = PlayOneMatch(args_match, player_one, player_two, chess_simulator, syzygy, pathDirectory)
        print('deterministic_behavior', global_variables.deterministic_behavior)

        if global_variables.profiling_bool:
            play.play_the_match()

        if not global_variables.profiling_bool:
            chessGui = QApplication(sys.argv)
            window = MainWindow(play)
            window.show()
            chessGui.exec_()
