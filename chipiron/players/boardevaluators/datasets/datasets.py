"""Document the module contains classes for handling datasets used in board evaluation tasks.

Classes:
- MyDataSet: A custom dataset class that loads and preprocesses data.
- FenAndValueDataSet: A subclass of MyDataSet that processes raw rows into input and target value tensors.

Functions:
- process_stockfish_value: A function that processes the stockfish value for a given board and row.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, no_type_check

import numpy as np
import pandas
import torch
from atomheart.board import IBoard
from atomheart.board.factory import create_board_chi
from atomheart.board.utils import Fen, FenPlusHistory
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from pandas import DataFrame
from torch.utils.data import Dataset

from chipiron.environments.chess.types import ChessState
from chipiron.utils import path
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from atomheart import BoardChi


@dataclass
class DataSetArgs:
    """Arguments for the dataset."""

    train_file_name: path
    test_file_name: path | None = None
    preprocessing_data_set: bool = False


type RawSample = pandas.Series


@no_type_check
class MyDataSet[ProcessedSample](Dataset[ProcessedSample], ABC):
    """A custom dataset class that loads and preprocesses data.

    Attributes:
    - file_name (str): The file name of the dataset.
    - preprocessing (bool): Flag indicating whether to preprocess the dataset.
    - data (pandas.DataFrame | list[ProcessedSample] | None): The loaded and processed data.
    - len (int | None): The length of the dataset.

    Methods:
    - load(): Loads the dataset from the file.
    - process_raw_row(row: pandas.Series) -> ProcessedSample: Processes a raw row into processed sample.

    """

    data: pandas.DataFrame | list[ProcessedSample] | None
    len: int | None
    preprocessing: bool

    def __init__(self, file_name: path, preprocessing: bool) -> None:
        """Initialize a new instance of the MyDataSet class.

        Args:
        - file_name (str): The file name of the dataset.
        - preprocessing (bool): Flag indicating whether to preprocess the dataset.

        """
        self.file_name = file_name
        self.preprocessing = preprocessing
        self.data = None

    @no_type_check
    def load(self) -> None:
        """Load the dataset from the file."""
        chipiron_logger.info("Loading the dataset...")
        start_time = time.time()

        raw_data_temp = pandas.read_pickle(self.file_name).copy()
        print("DEBUG columns", raw_data_temp.columns)
        print(
            "DEBUG first row types",
            {c: type(raw_data_temp.iloc[0][c]) for c in raw_data_temp.columns},
        )

        # Ensure we always work with a DataFrame
        raw_data: DataFrame
        if isinstance(raw_data_temp, pandas.Series):
            raw_data = raw_data_temp.to_frame()
        else:
            raw_data = raw_data_temp

        chipiron_logger.info("raw_data %s", str(type(raw_data).__name__))
        chipiron_logger.info(
            "--- LOAD READ PICKLE %.2f seconds ---", time.time() - start_time
        )
        chipiron_logger.info("Dataset loaded with %d items", len(raw_data))

        if self.preprocessing:
            chipiron_logger.info("Preprocessing dataset...")
            processed_data: list[ProcessedSample] = []

            # Handle DataFrame case
            for idx in range(len(raw_data)):
                if idx % 10 == 0:
                    chipiron_logger.info(
                        "Processing progress: %.2f%%", idx / len(raw_data) * 100
                    )
                row: pandas.Series[Any] = raw_data.iloc[idx]
                processed_data.append(self.process_raw_row(row))

            self.data = processed_data
            chipiron_logger.info("Preprocessing complete.")
        else:
            chipiron_logger.info("No preprocessing applied.")
            self.data = raw_data

        # Fix pandas pickle compatibility
        if isinstance(self.data, (pandas.DataFrame, list)):
            self.data = self.data.copy()

    @abstractmethod
    def process_raw_row(self, row: pandas.Series) -> ProcessedSample:
        """Convert a raw row into processed sample.

        Subclasses must implement this.
        """
        ...

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
        - int: The length of the dataset.

        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded yet. Call `load()` first.")
        return len(self.data)

    @no_type_check
    def __getitem__(self, idx: int) -> ProcessedSample:
        """Return the item at the given index.

        Always returns a ProcessedSample, either from preprocessed data or by processing on-the-fly.

        Args:
        - idx (int): The index of the item.

        Returns:
        - ProcessedSample: The processed sample.

        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded yet. Call `load()` first.")

        index = idx % len(self)
        if self.preprocessing:
            # Data is already preprocessed
            assert isinstance(self.data, list)
            return self.data[index]
        # Process on-the-fly
        assert isinstance(self.data, pandas.DataFrame)
        return self.process_raw_row(self.data.iloc[index])

    @no_type_check
    def get_unprocessed(self, idx: int) -> pandas.Series:
        """Return the unprocessed raw row at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            pandas.Series: The raw row.

        """
        if not isinstance(self.data, pandas.DataFrame):
            raise RuntimeError("Unprocessed data is not available.")
        return self.data.iloc[idx % len(self)]

    def is_preprocessed(self) -> bool:
        """Check if the dataset is preprocessed.

        Returns:
            bool: True if the dataset is preprocessed, False otherwise.

        """
        return self.preprocessing and isinstance(self.data, list)


@no_type_check
def process_stockfish_value(row: pandas.Series) -> float:
    """Process the stockfish value for a given board and row.

    Args:
    - row (pandas.Series): The row from the dataset.

    Returns:
    - float: The processed target value.

    """
    # target values are value between -1 and 1 from the point of view of white. (+1 is white win and -1 is white loose)
    target_value: float = np.tanh(row["stockfish_value"] / 500.0)
    return target_value


class SupervisedData(Protocol):
    """A protocol that defines the structure for classes that have input and target value attributes."""

    is_batch: bool = False  # Flag to indicate if this contains batched data

    def get_input_layer(self) -> torch.Tensor:
        """Return the input layer tensor."""
        ...

    def get_target_value(self) -> torch.Tensor:
        """Return the target value tensor."""
        ...


@dataclass
class FenAndValueData:
    """Represents the FEN and value data for a chess position."""

    fen_tensor: torch.Tensor
    value_tensor: torch.Tensor
    is_batch: bool = False  # Flag to indicate if this contains batched data

    def get_input_layer(self) -> torch.Tensor:
        """Return the input layer tensor."""
        return self.fen_tensor

    def get_target_value(self) -> torch.Tensor:
        """Return the target value tensor."""
        return self.value_tensor


def custom_collate_fn_fen_and_value(batch: list[FenAndValueData]) -> FenAndValueData:
    inputs = [item.get_input_layer() for item in batch]
    targets = [item.get_target_value() for item in batch]

    inputs_batch = torch.stack(inputs)
    targets_batch = torch.stack(targets)

    return FenAndValueData(
        fen_tensor=inputs_batch, value_tensor=targets_batch, is_batch=True
    )


class FenAndValueDataSet(MyDataSet[FenAndValueData]):
    """A subclass of MyDataSet that processes raw rows into input and target value tensors.

    Attributes:
    - transform_board_function (ContentToInputFunction): The function to transform the board into input tensor.
    - transform_value_function (callable): The function to transform the value for a given board and row.

    Methods:
    - process_raw_row(row: pandas.Series) -> FenAndValueData: Processes a raw row into FenAndValueData.
    - process_raw_rows(dataframe: pandas.DataFrame) -> list[FenAndValueData]: Processes raw rows into FenAndValueData list.

    """

    transform_board_function: ContentToInputFunction[
        ChessState
    ]  # transform board to model input
    transform_dataset_value_to_white_value_function: Callable[
        [pandas.Series], float
    ]  # transform value in dataset to standardized value white float
    transform_white_value_to_model_output_function: Callable[
        [float, IBoard], torch.Tensor
    ]  # transform white value to model output

    def __init__(
        self,
        file_name: path,
        transform_white_value_to_model_output_function: Callable[
            [float, IBoard], torch.Tensor
        ],
        transform_dataset_value_to_white_value_function: Callable[
            [pandas.Series], float
        ],
        transform_board_function: ContentToInputFunction[ChessState],
        preprocessing: bool = False,
    ) -> None:
        """Initialize a new instance of the FenAndValueDataSet class.

        Args:
        - file_name (str): The file name of the dataset.
        - preprocessing (bool): Flag indicating whether to preprocess the dataset.
        - transform_board_function (ContentToInputFunction): The function to transform the board into input tensor.
        - transform_dataset_value_to_white_value_function: The function to transform dataset values to white values.
        - transform_white_value_to_model_output_function: The function to transform white values to model output.

        """
        super().__init__(file_name, preprocessing)

        assert isinstance(transform_board_function, ContentToInputFunction)
        self.transform_board_function = transform_board_function

        # transform function
        self.transform_dataset_value_to_white_value_function = (
            transform_dataset_value_to_white_value_function
        )

        self.transform_white_value_to_model_output_function = (
            transform_white_value_to_model_output_function
        )

    @no_type_check
    def process_raw_row(self, row: pandas.Series) -> FenAndValueData:
        """Process a raw row into FenAndValueData.

        Args:
        - row (pandas.Series): The raw row from the dataset.

        Returns:
        - FenAndValueData: The processed data containing input and target tensors.

        """
        fen_: Fen = row["fen"]

        # todo should we make it general and allow rust boards just for testing all comptabilities ?
        board: BoardChi = create_board_chi(
            fen_with_history=FenPlusHistory(current_fen=fen_),
            use_board_modification=True,
        )

        input_layer: torch.Tensor = self.transform_board_function(ChessState(board))
        target_value_white: float = (
            self.transform_dataset_value_to_white_value_function(row)
        )
        target_value: torch.Tensor = (
            self.transform_white_value_to_model_output_function(
                target_value_white, board
            )
        )
        return FenAndValueData(fen_tensor=input_layer, value_tensor=target_value)

    def process_raw_rows(self, dataframe: pandas.DataFrame) -> list[FenAndValueData]:
        """Process raw rows into input and target tensors.

        Args:
        - dataframe (pandas.DataFrame): The raw rows from the dataset.

        Returns:
        - list[FenAndValueData]: The processed input and target tensors.

        """
        processed_data: list[FenAndValueData] = []
        row: pandas.Series
        for _, row in dataframe.iterrows():
            processed_data.append(self.process_raw_row(row))

        return processed_data
