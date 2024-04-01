import time
from typing import Any

import chess
import numpy as np
import pandas
import pandas as pd
import torch
from torch.utils.data import Dataset

from chipiron.environments.chess.board import BoardChi
from chipiron.environments.chess.board.factory import create_board
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import BoardToInputFunction


class MyDataSet(Dataset[Any]):
    data: pandas.DataFrame | list[tuple[torch.Tensor, torch.Tensor]] | None
    len: int | None

    def __init__(
            self,
            file_name: str,
            preprocessing: bool
    ) -> None:
        self.file_name = file_name
        self.preprocessing = preprocessing
        self.data = None
        self.len = None

    def load(self) -> None:
        print('Loading the dataset...')
        start_time = time.time()
        raw_data: pandas.DataFrame = pd.read_pickle(self.file_name)
        print('raw_data', type(raw_data))
        raw_data = raw_data.copy()  # gets read of compatibility problem between various version of panda and pickle
        print("--- LOAD READ PICKLE %s seconds ---" % (time.time() - start_time))
        print('Dataset  loaded with {} items'.format(len(raw_data)))

        # preprocessing
        if self.preprocessing:
            print('preprocessing dataset...')
            processed_data: list[tuple[torch.Tensor, torch.Tensor]] = []

            for idx in range(len(raw_data)):
                # print(idx, type(idx),idx % 10 == 0)
                if idx % 10 == 0:
                    print('\rloading the data', str(idx / len(raw_data) * 100) + '%')
                row: pandas.Series = raw_data.iloc[idx % len(raw_data)]
                processed_data.append(self.process_raw_row(row))
            self.data = processed_data

            print('preprocessing dataset done')
        else:
            print('no preprocessing the dataset')
            self.data = raw_data

        self.len = len(self.data)

    def process_raw_row(
            self,
            row: pandas.Series
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise Exception('should not be called')

    def __len__(self) -> int:
        assert (self.data is not None)
        return len(self.data)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | pandas.Series:
        assert (self.data is not None)

        if self.preprocessing:
            assert self.len is not None
            return self.data[idx % self.len]
        else:
            assert isinstance(self.data, pandas.DataFrame)
            assert self.len is not None

            raw_row = self.data.iloc[idx % self.len]
            return self.process_raw_row(raw_row)  # to be coded!


def process_stockfish_value(
        board: BoardChi,
        row: pandas.Series
) -> torch.Tensor:
    if board.turn == chess.BLACK:
        target_value = -np.tanh(row['stockfish_value'] / 500.)
    else:
        target_value = np.tanh(row['stockfish_value'] / 500.)
    target_value_tensor: torch.Tensor = torch.tensor([target_value])
    return target_value_tensor


class FenAndValueDataSet(MyDataSet):
    transform_board_function: BoardToInputFunction

    def __init__(
            self,
            file_name: str,
            preprocessing: bool = False,
            transform_board_function: str | BoardToInputFunction = 'identity',
            transform_value_function: str = ''
    ) -> None:

        super().__init__(file_name, preprocessing)
        # transform function
        if transform_board_function == 'identity':
            raise Exception(f'tobe coded in {__name__}')
        else:
            assert isinstance(transform_board_function, BoardToInputFunction)
            self.transform_board_function = transform_board_function

        # transform function
        if transform_value_function == 'stockfish':
            self.transform_value_function = process_stockfish_value

    def process_raw_row(
            self,
            row: pandas.Series
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fen = row['fen']
        board = create_board(fen=fen)
        input_layer = self.transform_board_function(board)
        target_value = self.transform_value_function(board, row)
        return input_layer.float(), target_value.float()

    def process_raw_rows(
            self,
            dataframe: pandas.DataFrame
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

        processed_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        row: pandas.Series
        for row in dataframe.iloc:
            processed_data.append(self.process_raw_row(row))

        return processed_data
