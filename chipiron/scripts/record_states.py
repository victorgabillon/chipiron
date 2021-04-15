import pandas as pd
import yaml
from src.games.play_one_game import PlayOneGame
from src.players import create_player
from src.chessenvironment.chess_environment import ChessEnvironment
from src.players.boardevaluators.syzygy import Syzygy
import global_variables


class RecordStates:

    def __init__(self):
        global_variables.deterministic_behavior = False
        global_variables.profiling_bool = False

        file_name_player_state_explorer = 'ZipfSequool.yaml'
        path_player_state_explorer = 'chipiron/runs/players/' + file_name_player_state_explorer
        exploration_power = 1000

        with open(path_player_state_explorer, 'r') as filePlayer_state_explorer:
            args_player_state_explorer = yaml.load(filePlayer_state_explorer, Loader=yaml.FullLoader)
            print('@~', args_player_state_explorer)

        args_player_state_explorer['tree_builder']['tree_move_limit'] = exploration_power

        file_game_name = 'setting_navo.yaml'
        path_game_setting = 'chipiron/runs/GameSettings/' + file_game_name

        with open(path_game_setting, 'r') as fileGame:
            self.args_game = yaml.load(fileGame, Loader=yaml.FullLoader)
            print(self.args_game)

        self.chess_simulator = ChessEnvironment()
        self.syzygy = Syzygy(self.chess_simulator,'')

        self.player_one = create_player(args_player_state_explorer, self.chess_simulator, self.syzygy)
        self.player_two = create_player(args_player_state_explorer, self.chess_simulator, self.syzygy)

        global_variables.init()  # global variables

        assert (not global_variables.deterministic_behavior)
        assert (not global_variables.profiling_bool)

    def run(self):

        count = 0
        while True:
            count += 1
            res = [0, 0, 0]
            list_of_new_rows = []
            for i in range(1):
                play = PlayOneGame(self.args_game, self.player_one, self.player_two, self.chess_simulator, self.syzygy)
                play.play_the_game()
                res[play.game.simple_results()] += 1
                print('@@@@@@@@@', res)
                for board in play.game.board_sequence:
                    new_row = {'fen': board.chess_board.fen()}
                    list_of_new_rows.append(new_row)

            print('@@@@@@@@@', res)
            data_frame_file_name = 'chipiron/data/states_from_given_policy/states_from_zipf_sequool.data'


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

