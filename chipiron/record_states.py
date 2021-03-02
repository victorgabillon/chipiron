import random
import pandas as pd
import yaml
from games.play_one_game import PlayOneGame
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings
import time
print('*8dddddddddddddddddddddddddddddddddddddddddddddddddd')
settings.deterministic_behavior = False
settings.profiling_bool = False

file_name_player_state_explorer = 'ZipfSequool.yaml'
path_player_state_explorer = 'chipiron/runs/players/' + file_name_player_state_explorer
only_game_over = True
classification_power = 250
exploration_power = 250

file_name_player_best_move_searcher = 'ZipfSequool.yaml'
path_player_best_move_searcher = 'chipiron/runs/players/' + file_name_player_best_move_searcher

with open(path_player_state_explorer, 'r') as filePlayer_state_explorer:
    args_player_state_explorer = yaml.load(filePlayer_state_explorer, Loader=yaml.FullLoader)
    print('@~', args_player_state_explorer)

args_player_state_explorer['tree_builder']['tree_move_limit'] = exploration_power

with open(path_player_best_move_searcher, 'r') as filePlayer_best_move_searcher:
    args_player_best_move_searcher = yaml.load(filePlayer_best_move_searcher, Loader=yaml.FullLoader)
    print('@d~', args_player_best_move_searcher)

args_player_best_move_searcher['tree_builder']['tree_move_limit'] = classification_power

file_game_name = 'setting_navo.yaml'
path_game_setting = 'chipiron/runs/GameSettings/' + file_game_name

with open(path_game_setting, 'r') as fileGame:
    args_game = yaml.load(fileGame, Loader=yaml.FullLoader)
    print(args_game)

chess_simulator = ChessEnvironment()
syzygy = Syzygy(chess_simulator)

player_one = create_player(args_player_state_explorer, chess_simulator, syzygy)
player_two = create_player(args_player_state_explorer, chess_simulator, syzygy)

player_best_move_searcher = create_player(args_player_best_move_searcher, chess_simulator, syzygy)

settings.init()  # global variables

assert (not settings.deterministic_behavior)
assert (not settings.profiling_bool)

count = 0
while True:
    count += 1
    res = [0, 0, 0]
    list_of_new_rows = []
    for i in range(10):
        play = PlayOneGame(args_game, player_one, player_two, chess_simulator, syzygy)
        play.play_the_game()
        res[play.game.simple_results()] += 1
        print('@@@@@@@@@', res)
        if only_game_over:
            selection = []
            for board in play.game.board_sequence:
                player_best_move_searcher.tree_explore(board)
                if player_best_move_searcher.tree.root_node.is_over():
                    print(board)
                    selection.append({'fen': board.chess_board.fen(),
                                      'final_value': player_best_move_searcher.tree.root_node.over_event.get_over_tag(),
                                      'explored': classification_power})
                    print({'fen': board.chess_board.fen(),
                           'final_value': player_best_move_searcher.tree.root_node.over_event.get_over_tag(),
                           'explored': classification_power})
            new_row = random.choice(selection)

        else:
            sample_board = random.choice(play.game.board_sequence)
            print('**', type(sample_board))
            # sample_board = play.game.chess_board_sequence[-1]
            new_row = {'fen': sample_board.chess_board.fen()}
        list_of_new_rows.append(new_row)

    print('@@@@@@@@@', res)
    data_frame_file_name = 'chipiron/data/states_Zs_softmax_50_num1.data'
    data_frame_file_name = 'chipiron/data/states_game_over_softmax_50_num2.data'

    #data_frame_file_name = 'chipiron/data/statestest0.data'

    try:
        data_frame_states = pd.read_pickle(data_frame_file_name)
    except:
        data_frame_states = None

    print('old_data_frame_states', data_frame_states)
    new_data_frame_states = pd.DataFrame.from_dict(list_of_new_rows)
    print('new_data_frame_states', new_data_frame_states)

    if data_frame_states is None:
        new_data_frame_states.to_pickle(data_frame_file_name)
        print('new_data_frame_states', new_data_frame_states)
    else:
        frames = [data_frame_states, new_data_frame_states]
        concat_data_frame_states = pd.concat(frames, ignore_index=True)
        print('concat_data_frame_states', concat_data_frame_states)
        print(type(concat_data_frame_states.iloc[0]['fen']))
        concat_data_frame_states.to_pickle(data_frame_file_name)

