"""
Module for converting the output of the neural network to a board evaluation
"""

from abc import ABC, abstractmethod

import chess
import torch

from chipiron.environments.chess.board import IBoard
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    FloatyBoardEvaluation,
)


class OutputValueConverter(ABC):
    """
    Converting an output of the neural network to a board evaluation
    and conversely converting a board evaluation to an output of the neural network
    """

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
    def from_value_white_to_model_output(
        self, board_value_white: float, board: IBoard
    ) -> torch.Tensor:
        """
        This functions takes the value white and converts to the corresponding value from the NN model output.
        Remember some NN models output value_from_mover for instance
        This function is used in training where the value white is compared to taget value from datasets that are float.
        """


# TODO This part should be reformated/improved as two concept are mixed
#  the convertion from the model output to  value white and the conversion to the final object used in pytorch as FloatyBoardEvaluation
#  maybe just rewrite a bit to make it less confusing
#  and the naming is not great as now we have multi option


class PlayerToMoveValueToValueWhiteConverter(OutputValueConverter):
    """
    Converting from a NN that outputs a 1D value from the point of view of the player to move
    """

    def convert_value_from_mover_viewpoint_to_value_white(
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
        value_white: float = self.convert_value_from_mover_viewpoint_to_value_white(
            turn=color_to_play, value_from_mover_view_point=value
        )
        board_evaluation: FloatyBoardEvaluation = FloatyBoardEvaluation(
            value_white=value_white
        )
        return board_evaluation

    def from_value_white_to_model_output(
        self, board_value_white: float, board: IBoard
    ) -> torch.Tensor:
        """
        This functions takes the value white and converts to the corresponding value from the NN model output.
        Remember some NN models output value_from_mover for instance
        This function is used in training where the value white is compared to taget value from datasets that are float.
        """
        value_from_mover_view_point: float
        if board.turn == chess.BLACK:
            value_from_mover_view_point = -board_value_white
        else:
            value_from_mover_view_point = board_value_white
        return torch.tensor([value_from_mover_view_point])


class IdentityConverter(OutputValueConverter):
    """
    Converting from a NN that outputs a 1D value from the point of view of the player to move
    """

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
        value_white: float = output_nn.item()
        board_evaluation: FloatyBoardEvaluation = FloatyBoardEvaluation(
            value_white=value_white
        )
        return board_evaluation

    def from_value_white_to_model_output(
        self, board_value_white: float, board: IBoard
    ) -> torch.Tensor:
        """
        This functions takes the value white and converts to the corresponding value from the NN model output.
        Remember some NN models output value_from_mover for instance
        This function is used in training where the value white is compared to taget value from datasets that are float.
        """
        return torch.tensor([board_value_white])
