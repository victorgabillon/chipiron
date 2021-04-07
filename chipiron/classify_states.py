import numpy as np
import yaml
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import global_variables
import pandas as pd
from chessenvironment.boards.board import MyBoard


def explore_and_update_df(data_frame_st, player, board_, index):
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


def syzygy_and_update_df(data_frame_st, board_, index_):
    if syzygy.fast_in_table(board_):
        # print('55',index_)
        data_frame_st.loc[index_, 'explored'] = 'syzygy'
        #print(board_,data_frame_st.loc[index_, 'fen'])
        data_frame_st.loc[index_, 'final_value'] = syzygy.get_over_tag(board_)


classification_power = 250
global_variables.deterministic_behavior = False
global_variables.profiling_bool = False
global_variables.learning_nn_bool = True

file_name_player_one = 'ZipfSequoolNN2.yaml'
path_player_one = 'chipiron/runs/players/' + file_name_player_one

with open(path_player_one, 'r') as filePlayerOne:
    args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
    print(args_player_one)


chess_simulator = ChessEnvironment()
syzygy = Syzygy(chess_simulator,'')

player_one = create_player(args_player_one, chess_simulator, syzygy)
assert (player_one.arg['tree_move_limit'] == classification_power)
# player_two = create_player(args_player_one, chess_simulator, syzygy)

global_variables.init()  # global variables

assert (not global_variables.deterministic_behavior)
assert (not global_variables.profiling_bool)

files = ['chipiron/data/states_good_from_png/subfile' + str(i) for i in range(10,50)]
files = ['chipiron/data/states_good_from_png/game_over_states_balanced_2']


for data_frame_file_name in files:
    print('the files is', data_frame_file_name)
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
    count = 0
    for index_, row_ in data_frame_states.iterrows():
        count += 1
        if not row_['explored'] == 'syzygy':
            if not row_['explored'] >= classification_power or (
                    row_['best_next_fen'] == np.NaN and row_['final_value'] == np.NaN):
                if count % 100 == 0:
                  print('^^', count, len(data_frame_states.index))
                fen = row_['fen']
                board = MyBoard(fen=fen)
                #explore_and_update_df(data_frame_states, player_one, board, index_)
                syzygy_and_update_df(data_frame_states,board,index_)

    print('old_data_frame_states', data_frame_states)
    data_frame_states.to_pickle(data_frame_file_name)
