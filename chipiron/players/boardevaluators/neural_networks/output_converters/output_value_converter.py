"""
Module for converting the output of the neural network to a board evaluation
"""

from abc import ABC, abstractmethod

import chess
import torch

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    FloatyBoardEvaluation,
    PointOfView,
)


class OutputValueConverter(ABC):
    """
    Converting an output of the neural network to a board evaluation
    and conversely converting a board evaluation to an output of the neural network
    """

    point_of_view: PointOfView

    def __init__(self, point_of_view: PointOfView) -> None:
        """
        Initialize the OutputValueConverter with a given point of view.

        Args:
            point_of_view (PointOfView): The point of view for the conversion.
        """
        self.point_of_view = point_of_view

    @abstractmethod
    def to_board_evaluation(
        self, output_nn: torch.Tensor, color_to_play: chess.Color
    ) -> FloatyBoardEvaluation:
        """
        Convert the output of the neural network to a board evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            color_to_play (chess.Color): The color of the player to move.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.
        """
        ...

    @abstractmethod
    def to_nn_outputs(self, board_evaluation: FloatyBoardEvaluation) -> torch.Tensor:
        """
        Convert a board evaluation to the output of the neural network.

        Args:
            board_evaluation (FloatyBoardEvaluation): The board evaluation to convert.

        Returns:
            torch.Tensor: The converted output of the neural network.
        """
        ...


class OneDToValueWhite(OutputValueConverter):
    """
    Converting from a NN that outputs a 1D value from the point of view of the player to move
    """

    def convert_value_for_mover_viewpoint_to_value_white(
        self, turn: chess.Color, value_from_mover_view_point: float
    ) -> float:
        """
        Convert the value from the mover's viewpoint to the value from the white player's viewpoint.

        Args:
            turn (chess.Color): The color of the player to move.
            value_from_mover_view_point (float): The value from the mover's viewpoint.

        Returns:
            float: The value from the white player's viewpoint.
        """
        if turn == chess.BLACK:
            value_white = -value_from_mover_view_point
        else:
            value_white = value_from_mover_view_point
        return value_white

    def to_board_evaluation(
        self, output_nn: torch.Tensor, color_to_play: chess.Color
    ) -> FloatyBoardEvaluation:
        """
        Convert the output of the neural network to a board evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            color_to_play (chess.Color): The color of the player to move.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.
        """
        value: float = output_nn.item()
        value_white: float = self.convert_value_for_mover_viewpoint_to_value_white(
            turn=color_to_play, value_from_mover_view_point=value
        )
        board_evaluation: FloatyBoardEvaluation = FloatyBoardEvaluation(
            value_white=value_white
        )
        return board_evaluation

    def to_nn_outputs(self, board_evaluation: FloatyBoardEvaluation) -> torch.Tensor:
        """
        Convert a board evaluation to the output of the neural network.

        Args:
            board_evaluation (FloatyBoardEvaluation): The board evaluation to convert.

        Returns:
            torch.Tensor: The converted output of the neural network.
        """
        raise Exception("Not implemented in output_value_converter.py")
