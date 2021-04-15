from src.players.boardevaluators.neural_networks.NN4_pytorch import NN4Pytorch, transform_board
import pandas as pd
from src.chessenvironment.boards.board import MyBoard
import chess
import torch
import numpy as np
import math

nn_param_file_name = 'NN104/paramtest_1145.pt'

nn = NN4Pytorch(nn_param_file_name)

# df = pd.read_pickle('chipiron/data/states_good_from_png/subsub0')
df = pd.read_pickle('../data/states/random_games')
df_over = pd.read_pickle('chipiron/data/states_good_from_png/game_over_states_balanced_2')
sum_input = torch.zeros(10)

for i in range(100 * len(df.index)):
    if i % 2 == 0:
        one_row_df = df_over.sample()
    else:
        one_row_df = df.sample()

    #  one_row_df = df_over.sample()

    row = one_row_df.iloc[0]
    print(row, type(row))
    fen = row['fen']
    final_value = row['final_value']
    board = MyBoard(fen=fen)
    input_layer = transform_board(board, requires_grad_=False)
    print(board.chess_board, final_value, row['final_value'], input_layer)

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
        print('final value is ', final_value, type(final_value), final_value == np.NaN, math.isnan(final_value))
        raise Exception('no!!!')

    #    assert(3==4)
    print(input_layer, target_value, target_input_layer)

    nn.train_one_example(input_layer, target_value, target_input_layer)
