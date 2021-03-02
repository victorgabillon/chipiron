import numpy as np
import yaml
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings
import pandas as pd
from chessenvironment.boards.board import MyBoard


def explore_and_update_df(data_frame_st, player, board_,index):
    player.tree_explore(board_)
    data_frame_st.loc[index, 'explored'] = classification_power
    if player.tree.root_node.is_over():
        data_frame_st.loc[index, 'final_value'] = player.tree.root_node.over_event.get_over_tag()
    if player.tree.root_node.board.chess_board.is_game_over():
        print('~~~~')
    else:
        if syzygy.fast_in_table(board_):
            best_move = syzygy.best_move(board_)
            best_next_board = chess_simulator.step_create(board_, best_move, 0)
            best_next_fen = best_next_board.chess_board.fen()
        else:
            print('@@', board_)
            best_child_node = player.tree.root_node.best_child()
            best_next_fen = best_child_node.board.chess_board.fen()
        data_frame_st.loc[index, 'best_next_fen'] = best_next_fen

def syzygy_and_update_df(data_frame_st, board_,index):

        if syzygy.fast_in_table(board_):
            syzygy.val(board_)
            data_frame_st.loc[index, 'explored'] = 'syzygy'
            data_frame_st.loc[index, 'final_value'] = best_next_fen


classification_power = 250
settings.deterministic_behavior = False
settings.profiling_bool = False
settings.learning_nn_bool = True

file_name_player_one = 'ZipfSequoolNN2.yaml'
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

data_frame_file_name = 'chipiron/data/states_from_png.data'
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

assert (not settings.deterministic_behavior)
assert (not settings.profiling_bool)

for index_, row_ in data_frame_states.iterrows():
    if not row_['explored'] >= classification_power or (row_['best_next_fen'] == np.NaN and row_['final_value'] == np.NaN):
        print('^^')
        fen = row_['fen']
        board = MyBoard(fen=fen)
        explore_and_update_df(data_frame_states, player_one, board,index_)

print('old_data_frame_states', data_frame_states)
print(data_frame_states.loc[1, 'fen'])
data_frame_states.to_pickle(data_frame_file_name)
