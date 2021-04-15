from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import math
import chess
from src.chessenvironment.boards.board import MyBoard


class FenDataSet(Dataset):

    def __init__(self, transform_function='identity'):
        if transform_function == 'identity':
            self.transform_function = lambda x: x
        else:
            self.transform_function = transform_function


class ClassifiedBoards(FenDataSet):

    def __init__(self, transform_function):
        super().__init__(transform_function)
        print('Loading the ClassifiedBoards dataset...')
        self.df_over = pd.read_pickle('chipiron/data/states_random/game_over_states')
        self.df_over_2 = pd.read_pickle('/home/victor/good_games_game_over_classified')
        self.df_over_3 = pd.read_pickle('/home/victor/random_classify_merge')
        print('Dataset ClassifiedBoards loaded.')
        self.len = len(self.df_over)
        self.len_2 = len(self.df_over_2)
        self.len_3 = len(self.df_over_3)

    def __len__(self):
        return len(self.df_over) #+ len(self.df_over_2)  # + len(self.df_over_3)

    def __getitem__(self, idx):
        if random.random() > .8:
            row = self.df_over.iloc[idx % self.len]
        elif random.random() > .4:
            row = self.df_over_2.iloc[idx % self.len_2]
        else:
             row = self.df_over_3.iloc[idx % self.len_3]

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


class NextBoards(FenDataSet):

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
