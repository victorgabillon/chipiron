"""
Module for the BoardToInput protocol and BoardToInputFunction protocol.
"""
from typing import Any
from typing import Protocol, runtime_checkable

import torch

import chipiron.environments.chess.board as boards


class BoardToInput(Protocol):
    """
    Protocol for converting a chess board to a tensor input for a neural network.
    """

    def convert(
            self,
            board: boards.IBoard[Any]
    ) -> torch.Tensor:
        """
        Converts the given chess board to a tensor input.

        Args:
            board (BoardChi): The chess board to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...


@runtime_checkable
class BoardToInputFunction(Protocol):
    """
    Protocol for a callable object that converts a chess board to a tensor input for a neural network.
    """

    def __call__(
            self,
            board: boards.IBoard[Any]
    ) -> torch.Tensor:
        """
        Converts the given chess board to a tensor input.

        Args:
            board (BoardChi): The chess board to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...
