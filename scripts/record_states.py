import pandas as pd
import yaml
from chipiron.players.player import Player
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
import os

class RecordStates:

    def __init__(self):

        self.algo_name = 'RecurZipfBase'
        self.tree_moves = 1000
        self.temperature = 14

        file_name_player_state_explorer = self.algo_name + '.yaml'
        path_player_state_explorer = 'chipiron/runs/players/best_players/explore_bests/RecurZipfBase.yaml'# + file_name_player_state_explorer

        with open(path_player_state_explorer, 'r') as filePlayer_state_explorer:
            args_player_state_explorer = yaml.load(filePlayer_state_explorer, Loader=yaml.FullLoader)
            print('@~', args_player_state_explorer)

        args_player_state_explorer['tree_builder']['tree_move_limit'] = self.tree_moves
        args_player_state_explorer['tree_builder']['move_selection_rule']['temperature'] = self.temperature

        file_game_name = 'setting_navo.yaml'
        path_game_setting = 'chipiron/runs/GameSettings/' + file_game_name

        with open(path_game_setting, 'r') as fileGame:
            self.args_game = yaml.load(fileGame, Loader=yaml.FullLoader)
            print(self.args_game)

        self.chess_simulator = ChessEnvironment()
        self.syzygy = SyzygyTable(self.chess_simulator, '')

        self.player_one = Player(args_player_state_explorer, self.chess_simulator, self.syzygy)
        self.player_two = Player(args_player_state_explorer, self.chess_simulator, self.syzygy)

        global_variables.init()  # global variables

        self.folder_name = 'chipiron/data/states/states_from_fixed_policy/' + self.algo_name + '-' + str(
            self.tree_moves) + '-' + str(self.temperature)
        os.mkdir(self.folder_name)
        self.data_state_filname = self.folder_name + 'states.data'
        player_filename = self.folder_name + '/player_info.yaml'
        with open(player_filename, 'w') as out_file:
            out_file.write(yaml.dump(self.player_one.arg))
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

            try:
                data_frame_states = pd.read_pickle(self.data_state_filname)
            except:
                data_frame_states = None

            print('old_data_frame_states', data_frame_states)
            new_data_frame_states = pd.DataFrame.from_dict(list_of_new_rows)
            print('new_data_frame_states', new_data_frame_states)

            if data_frame_states is None:
                new_data_frame_states.to_pickle(self.data_state_filname)
                print('new_data_frame_states', new_data_frame_states)
            else:
                frames = [data_frame_states, new_data_frame_states]
                concat_data_frame_states = pd.concat(frames, ignore_index=True)
                print('concat_data_frame_states', concat_data_frame_states)
                print(type(concat_data_frame_states.iloc[0]['fen']))
                concat_data_frame_states.to_pickle(self.data_state_filname)

