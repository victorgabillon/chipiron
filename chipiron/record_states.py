
import random
import pandas as pd

import yaml
from games.play_one_game import PlayOneGame
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings

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
player_two = create_player(args_player_one, chess_simulator, syzygy)

settings.init()  # global variables

list_of_new_rows = []

for i in range(10):
    play = PlayOneGame(args_game, player_one, player_two, chess_simulator, syzygy)
    play.play_the_game()
    sample_board = random.choice(play.game.chess_board_sequence)
    print('**',type(sample_board))
    # sample_board = play.game.chess_board_sequence[-1]
    new_row = {'board': sample_board}
    list_of_new_rows.append(new_row)

data_frame_file_name = 'chipiron/data/states.data'
try:
    data_frame_states = pd.read_pickle(data_frame_file_name)
except:
    data_frame_states = None

print('old_data_frame_states', data_frame_states)
new_data_frame_states = pd.DataFrame.from_dict(list_of_new_rows)

if data_frame_states is None:
    new_data_frame_states.to_pickle(data_frame_file_name)
    print('new_data_frame_states', new_data_frame_states)
else:
    frames = [data_frame_states, new_data_frame_states]
    concat_data_frame_states = pd.concat(frames, ignore_index=True)
    print('concat_data_frame_states', concat_data_frame_states)
    print(type(concat_data_frame_states.iloc[0]['board']))
    concat_data_frame_states.to_pickle(data_frame_file_name)
