from players.boardevaluators.NN4_pytorch import NN4Pytorch, transform_board
import pandas as pd
from chessenvironment.boards.board import MyBoard
import chess
import torch
import numpy as np
import math

folder = 'NN104'

nn = NN4Pytorch(folder)

df = pd.read_pickle('chipiron/data/test50_2')

sum_input = torch.zeros(10)

for i in range(100 * len(df.index)):
    one_row_df = df.sample()
    row = one_row_df.iloc[0]
    print(row, type(row))
    fen = row['fen']
    final_value = row['final_value']
    board = MyBoard(fen=fen)
    input_layer = transform_board(board, requires_grad_=False)
    print(board.chess_board,final_value,row['final_value'],input_layer)

    target_input_layer = None
    if final_value == 'Win-Wh':
        if board.chess_board.turn == chess.WHITE:
            target_value_0_1 = 1
        else:
            target_value_0_1 = 0
    elif final_value == 'Win-Bl':
        if board.chess_board.turn == chess.WHITE:
            target_value_0_1 = 0
        else:
            target_value_0_1 = 1
    elif final_value == 'Draw':
        target_value_0_1 = .5
    elif math.isnan(final_value):
        target_value_0_1 = None
        next_fen = row['best_next_fen']
        next_board = MyBoard(fen=next_fen)
        target_input_layer = transform_board(next_board, requires_grad_=False)
        print('56',next_board,target_input_layer)

    else:
        print('final value is ',final_value,type(final_value),final_value==np.NaN,math.isnan(final_value))
        raise Exception('no!!!')


#    assert(3==4)
    print(input_layer, target_value_0_1, target_input_layer)
    # sum_input = sum_input + input_layer
    # if input_layer[1] > 0:
    #     cav += 1 / (input_layer[1]) * (2 * target_value_0_1 - 1)
    # if input_layer[2] > 0:
    #     bis += 1 / input_layer[2] * (2 * target_value_0_1 - 1)
    # if input_layer[0] > 0:
    #     paw += 1 / input_layer[0] * (2 * target_value_0_1 - 1)
    #
    # print('##', type(input_layer), sum_input, input_layer, input_layer[1], cav, bis, paw)

    nn.train_one_example(input_layer, target_value_0_1, target_input_layer)
