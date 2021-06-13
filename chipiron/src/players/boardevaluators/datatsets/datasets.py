import time

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math
import chess
from src.chessenvironment.boards.board import MyBoard
import torch



class MyDataSet(Dataset):

    def __init__(self, file_name, preprocessing):
        self.file_name = file_name
        self.preprocessing = preprocessing
        self.data = None
        self.len = None

    def load(self):
        print('Loading the dataset...')
        raw_data = pd.read_pickle(self.file_name)
        print('Dataset  loaded.')

        # preprocessing
        if self.preprocessing:
            print('preprocessing dataset...')
            processed_data = []
            for idx in range(len(raw_data)):
                print(idx, len(raw_data))
                row = raw_data.iloc[idx % len(raw_data)]
                processed_data.append(self.process_raw_row(row))
            self.data = processed_data
            print('preprocessing dataset done')
        else:
            print('no preprocessing the dataset')
            self.data = raw_data


        self.len = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preprocessing:
            return self.data[idx % self.len]
        else:
            raw_row = self.data.iloc[idx % self.len]
            return self.process_raw_row(raw_row)  # to be coded!


def process_stockfish_value(board, row):
    if board.chess_board.turn == chess.BLACK:
        target_value = -np.tanh(row['stockfish_value'] / 500.)
    else:
        target_value = np.tanh(row['stockfish_value'] / 500.)
    target_value = torch.tensor([target_value])
    return target_value


class FenAndValueDataSet(MyDataSet):

    def __init__(self, file_name, preprocessing=False, transform_board_function='identity',
                 transform_value_function=''):

        super().__init__( file_name, preprocessing)
        # transform function
        if transform_board_function == 'identity':
            assert (1 == 0)  # to be coded
        else:
            self.transform_board_function = transform_board_function

        # transform function
        if transform_value_function == 'stockfish':
            self.transform_value_function = process_stockfish_value

    def process_raw_row(self, row):
        fen = row['fen']
        board = MyBoard(fen=fen)
        input_layer = self.transform_board_function(board, requires_grad_=False)
        target_value = self.transform_value_function(board, row)
        return input_layer, target_value



class ClassifiedBoards(MyDataSet):

    def __init__(self, transform_function):
        super().__init__(transform_function)
        print('Loading the ClassifiedBoards dataset...')
        self.df_over = pd.read_pickle('chipiron/data/states_random/game_over_states')
        self.df_over_2 = pd.read_pickle('/home/victor/good_games_classified_stock_withmoredraws')
        self.df_over_3 = pd.read_pickle('/home/victor/random_stockfish_classify_merge_bal')
        self.df_over_4 = pd.read_pickle('/home/victor/good_stockfish_classify_merge_bal')
        print('Dataset ClassifiedBoards loaded.')
        self.len = len(self.df_over)
        self.len_2 = len(self.df_over_2)
        self.len_3 = len(self.df_over_3)
        self.len_4 = len(self.df_over_4)

    def __len__(self):
        return len(self.df_over)  # + len(self.df_over_2)  # + len(self.df_over_3)

    def __getitem__(self, idx):
        row = self.df_over_4.iloc[idx % self.len_4]

        #     if random.random() > .8:
        #         row = self.df_over.iloc[idx % self.len]
        #     elif random.random() > .4:
        #         row = self.df_over_2.iloc[idx % self.len_2]
        #     else:
        #         row = self.df_over_3.iloc[idx % self.len_3]
        # #        print('^^', idx % self.len_3, self.len_3,row)

        # row = self.df_over.iloc[idx]
        fen = row['fen']
        # print('fen',fen)
        final_value = row['final_value']
        board = MyBoard(fen=fen)
        # print(board)
        input_layer = self.transform_function(board, requires_grad_=False)
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
        else:
            print('final value is ', final_value, type(final_value), final_value == np.NaN,
                  math.isnan(final_value))
            raise Exception('no!!!')

        return input_layer, target_value


class NextBoards(MyDataSet):

    def __init__(self, transform_function):
        super().__init__(transform_function)
        print('Loading the NextBoards dataset...')
        self.df_next = pd.read_pickle('/home/victor/next_board_dataset_from good_games_0')
        print('Dataset NextBoards loaded.')

    def __len__(self):
        return len(self.df_next)

    def __getitem__(self, idx):
        row = self.df_next.iloc[idx]
        fen = row['fen']
        next_fen = row['next_fen']
        board = MyBoard(fen=fen)
        input_layer = self.transform_function(board, requires_grad_=False)
        next_board = MyBoard(fen=next_fen)
        # print(fen)
        # print(board)
        # print(next_fen)
        # print( next_board )
        next_input_layer = self.transform_function(next_board, requires_grad_=False)
        return input_layer, next_input_layer


class States(Dataset):

    def __init__(self):
        print('Loading the States dataset...')
        self.df_next = pd.read_pickle('chipiron/datat')
        print('Dataset States loaded.')

    def __len__(self):
        return len(self.df_next)

    def __getitem__(self, idx):
        row = self.df_next.iloc[idx]
        fen = row['fen']
        next_fen = row['next_fen']
        board = MyBoard(fen=fen)
        input_layer = transform_board_pieces_square(board, requires_grad_=False)
        next_board = MyBoard(fen=next_fen)
        # print(fen)
        # print(board)
        # print(next_fen)
        # print( next_board )
        next_input_layer = transform_board_pieces_square(next_board, requires_grad_=False)
        return input_layer, next_input_layer
