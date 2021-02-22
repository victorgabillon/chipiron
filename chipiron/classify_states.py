import numpy as np
import yaml
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings
import pandas as pd
from chessenvironment.boards.board import MyBoard

classification_power = 250
settings.deterministic_behavior = False
settings.profiling_bool = False
settings.learning_nn_bool = True

file_name_player_one = 'ZipfSequool.yaml'
path_player_one = 'chipiron/runs/players/' + file_name_player_one

with open(path_player_one, 'r') as filePlayerOne:
    args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
    print(args_player_one)

file_game_name = 'setting_navo.yaml'
path_game_setting = 'chipiron/runs/GameSettings/' + file_game_name

with open(path_game_setting, 'r') as fileGame:
    args_game = yaml.load(fileGame, Loader=yaml.FullLoader)
    print(args_game)

chess_simulator = ChessEnvironment()
syzygy = Syzygy(chess_simulator)

player_one = create_player(args_player_one, chess_simulator, syzygy)
assert (player_one.arg['tree_move_limit'] == classification_power)
# player_two = create_player(args_player_one, chess_simulator, syzygy)

settings.init()  # global variables

data_frame_file_name = 'chipiron/data/states2.data'
try:
    data_frame_states = pd.read_pickle(data_frame_file_name)
except:
    data_frame_states = None

if 'explored' not in data_frame_states:
    data_frame_states['explored'] = np.NaN

if 'final_value' not in data_frame_states:
    data_frame_states['final_value'] = np.NaN

if 'best_next_fen' not in data_frame_states:
    data_frame_states['best_next_fen'] = np.NaN

for index, row in data_frame_states.iterrows():
    if not row['explored'] >= classification_power or row['best_next_board'] == np.NaN:
        fen = row['fen']
        board = MyBoard(fen=fen)
        player_one.tree_explore(board)
        data_frame_states.loc[index, 'explored'] = classification_power
        if player_one.tree.root_node.is_over():
            data_frame_states.loc[index, 'final_value'] = player_one.tree.root_node.over_event.simple_string()
        if player_one.tree.root_node.board.chess_board.is_game_over():
            print('~~~~')
        else:
            if syzygy.fast_in_table(board):
                best_move = syzygy.best_move(board)
                best_next_board = chess_simulator.step_create(board, best_move, 0)
            else:
                print('@@',board)
                best_child_node = player_one.tree.root_node.best_child()
                best_next_board = best_child_node.board
            data_frame_states.loc[index, 'best_next_fen'] = best_next_board

print('old_data_frame_states', data_frame_states)
print(data_frame_states.loc[1, 'fen'])
data_frame_states.to_pickle(data_frame_file_name)
