import pandas as pd
import numpy as np
import yaml
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import settings
from players.create_player import create_player


class LearnAndClassifyScript:

    def __init__(self):
        self.classification_power = 250
        settings.deterministic_behavior = False
        settings.profiling_bool = False
        settings.learning_nn_bool = True

        file_name_player_one = 'ZipfSequoolNN2.yaml'
        path_player_one = 'chipiron/runs/players/' + file_name_player_one

        with open(path_player_one, 'r') as filePlayerOne:
            args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player_one)

        chess_simulator = ChessEnvironment()
        syzygy = Syzygy(chess_simulator, '')

        player = create_player(args_player_one, chess_simulator, syzygy)
        assert (player.arg['tree_move_limit'] == self.classification_power)

        self.nn = player.board_evaluators_wrapper.board_evaluator

        # df = pd.read_pickle('chipiron/data/states_good_from_png/subsub0')
        df = pd.read_pickle('../data/states/random_games')
        if 'explored' not in df:
            df['explored'] = np.NaN
        if 'final_value' not in df:
            df['final_value'] = np.NaN
        if 'best_next_fen' not in df:
            df['best_next_fen'] = np.NaN
        df_dict = df.to_dict('index')

        df_over = pd.read_pickle('../data/states_random/game_over_states')
        df_over_2 = pd.read_pickle('/home/victor/oo')
        df_over_3 = pd.read_pickle('/home/victor/random_classify')



    def run(self):