import sys
import yaml
from games.play_one_match import PlayOneMatch
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings
from players.boardevaluators.NN4_pytorch import NN4Pytorch
from displays.gui import MainWindow
from PyQt5.QtWidgets import *
from data.replay_buffer import ReplayBufferManager

settings.deterministic_behavior = False
settings.profiling_bool = False
settings.learning_nn_bool = True

folder = 'NN104'
settings.nn_to_train = NN4Pytorch(folder)
settings.nn_replay_buffer_manager = ReplayBufferManager(folder)

file_name_player_one = 'ZipfSequoolNN2.yaml'
file_name_match_setting = 'setting_nero.yaml'
path_player_one = 'runs/players/' + file_name_player_one
path_match_setting = 'runs/OneMatch/' + file_name_match_setting
with open(path_match_setting, 'r') as fileMatch:
    args_match = yaml.load(fileMatch, Loader=yaml.FullLoader)
    print(args_match)

with open(path_player_one, 'r') as filePlayerOne:
    args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
    print(args_player_one)

fileGame = args_match['game_setting_file']
path_game_setting = 'runs/GameSettings/' + fileGame

chess_simulator = ChessEnvironment()
syzygy = Syzygy(chess_simulator)

player_one = create_player(args_player_one, chess_simulator, syzygy)
player_two = create_player(args_player_one, chess_simulator, syzygy)

play = PlayOneMatch(args_match, player_one, player_two, chess_simulator, syzygy)
settings.init()  # global variables

chessGui = QApplication(sys.argv)
window = MainWindow(play)
window.show()
chessGui.exec_()
