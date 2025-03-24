"""
This module provides a class for converting a chess board into a tensor representation using a 364-dimensional input.

Classes:
- Representation364BTI: Converts a chess board into a tensor representation.

"""

from typing import Any

import torch

import chipiron.environments.chess.board as boards

from .board_representation import BoardRepresentation
from .factory import RepresentationFactory


class RepresentationBTI:
    """
    Converts a chess board into a tensor representation using a 364-dimensional input.

    Methods:
    - __init__: Initializes the Representation364BTI object.
    - convert: Converts the chess board into a tensor representation.

    """

    def __init__(self, representation_factory: RepresentationFactory[Any]):
        """
        Initializes the Representation364BTI object.

        Parameters:
        - representation_factory (Representation364Factory): The factory object for creating the board representation.

        """
        self.representation_factory = representation_factory

    def convert(self, board: boards.IBoard) -> torch.Tensor:
        """
        Converts the chess board into a tensor representation.

        Parameters:
        - board (BoardChi): The chess board to convert.

        Returns:
        - tensor (torch.Tensor): The tensor representation of the chess board.

        """
        representation: BoardRepresentation = (
            self.representation_factory.create_from_board(board=board)
        )
        tensor: torch.Tensor = representation.get_evaluator_input(
            color_to_play=board.turn
        )
        return tensor.float()
