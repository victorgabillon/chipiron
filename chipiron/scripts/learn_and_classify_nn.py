import pandas as pd
import numpy as np
import yaml
from chessenvironment.chess_environment import ChessEnvironment
from players.boardevaluators.syzygy import Syzygy
import global_variables
from players.create_player import create_player
from players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
import torch


class LearnAndClassifyScript:

    def __init__(self):
        self.classification_power = 250
        global_variables.deterministic_behavior = False
        global_variables.profiling_bool = False
        global_variables.learning_nn_bool = True

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
        df = pd.read_pickle('chipiron/data/states/random_games')
        if 'explored' not in df:
            df['explored'] = np.NaN
        if 'final_value' not in df:
            df['final_value'] = np.NaN
        if 'best_next_fen' not in df:
            df['best_next_fen'] = np.NaN
        df_dict = df.to_dict('index')

        nn_trainer = NNPytorchTrainer(self.nn )

        df_over = pd.read_pickle('chipiron/data/states_random/game_over_states')
        df_over_2 = pd.read_pickle('/home/victor/oo')
        df_over_3 = pd.read_pickle('/home/victor/random_classify')

    def run(self):

        sum_input = torch.zeros(10)

        for i in range(100 * len(df.index)):
            if i % 10 > 10:
                print('################de')
                # one_row_df = df_over.sample()
                key, row_ = random.choice(list(df_dict.items()))
                fen = row_['fen']
                final_value = row_['final_value']
                board_ = MyBoard(fen=fen)
                print('fen', fen)
                print('%%', board_, board_.chess_board.is_valid())

                input_layer = transform_board(board_, requires_grad_=False)
                if not row_['explored'] == 'syzygy':
                    if not row_['explored'] >= classification_power or (
                            row_['best_next_fen'] == np.NaN and row_['final_value'] == np.NaN):
                        df_dict[key]['explored'] = classification_power
                        player.tree_explore(board_)
                        if player.tree.root_node.is_over():
                            df_dict[key]['final_value'] = player.tree.root_node.over_event.get_over_tag()
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
                                best_next_board = best_child_node.board
                                best_next_fen = best_child_node.board.chess_board.fen()
                            df_dict[key]['best_next_fen'] = best_next_fen

                            target_value = None
                            target_input_layer = transform_board(best_next_board, requires_grad_=False)
                            nn.train_one_example(input_layer, target_value, target_input_layer)



            else:
                if random.random() > .7:
                    # one_row_df = df.sample()
                    one_row_df = df_over.sample()
                elif random.random() > .4:
                    one_row_df = df_over_2.sample()
                else:
                    one_row_df = df_over_3.sample()

                row = one_row_df.iloc[0]
                # print(row, type(row))
                fen = row['fen']
                final_value = row['final_value']
                board = MyBoard(fen=fen)
                input_layer = transform_board(board, requires_grad_=False)
                # print(board.chess_board, final_value, row['final_value'], input_layer)

                target_input_layer = None
                if final_value == 'Win-Wh':
                    if board.chess_board.turn == chess.WHITE:
                        target_value = 1
                    else:
                        target_value = -1
                elif final_value == 'Win-Bl':
                    if board.chess_board.turn == chess.WHITE:
                        target_value = -1
                    else:
                        target_value = 1
                elif final_value == 'Draw':
                    target_value = 0
                elif math.isnan(final_value):
                    target_value = None
                    next_fen = row['best_next_fen']
                    next_board = MyBoard(fen=next_fen)
                    target_input_layer = transform_board(next_board, requires_grad_=False)
                    print('56', next_board, target_input_layer)

                else:
                    print('final value is ', final_value, type(final_value), final_value == np.NaN,
                          math.isnan(final_value))
                    raise Exception('no!!!')

                #    assert(3==4)
                # print(input_layer, target_value, target_input_layer)
                #
                nn.train_one_example(input_layer, target_value, target_input_layer)
