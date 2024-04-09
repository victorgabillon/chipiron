"""
Module defining the board representation interface and the 364 features board representation.
"""

from dataclasses import dataclass
from typing import Protocol

import chess
import torch


class BoardRepresentation(Protocol):
    """
    Protocol defining the interface for a board representation.
    """

    def get_evaluator_input(
            self,
            color_to_play: chess.Color
    ) -> torch.Tensor:
        """
        Returns the evaluator input tensor for the given color to play.

        Args:
            color_to_play: The color to play, either chess.WHITE or chess.BLACK.

        Returns:
            The evaluator input tensor.
        """
        ...


@dataclass(slots=True)
class Representation364(BoardRepresentation):
    """
    Dataclass representing a board representation with 364 features.
    """

    tensor_white: torch.Tensor
    tensor_black: torch.Tensor
    tensor_castling_white: torch.Tensor
    tensor_castling_black: torch.Tensor

    def get_evaluator_input(
            self,
            color_to_play: chess.Color
    ) -> torch.Tensor:
        """
        Returns the evaluator input tensor for the given color to play.

        Args:
            color_to_play: The color to play, either chess.WHITE or chess.BLACK.

        Returns:
            The evaluator input tensor.
        """
        if color_to_play == chess.WHITE:
            tensor = torch.cat((self.tensor_white, self.tensor_black), 0)
        else:
            tensor = torch.cat((self.tensor_black, self.tensor_white), 0)

        if color_to_play == chess.WHITE:
            tensor_castling = torch.cat((self.tensor_castling_white, self.tensor_castling_black), 0)
        else:
            tensor_castling = torch.cat((self.tensor_castling_black, self.tensor_castling_white), 0)

        tensor_2 = torch.cat((tensor, tensor_castling), 0)

        return tensor_2
