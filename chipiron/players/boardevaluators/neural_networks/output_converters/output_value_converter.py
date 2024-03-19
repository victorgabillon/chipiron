from abc import ABC, abstractmethod

import chess
import torch

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation, PointOfView


class OutputValueConverter(ABC):
    """
    Converting an output of the neural network to a board evaluation
     and conversely converting a board evaluation to an output of the neural network
    """

    point_of_view: PointOfView

    def __init__(
            self,
            point_of_view: PointOfView
    ) -> None:
        self.point_of_view = point_of_view

    @abstractmethod
    def to_board_evaluation(
            self,
            output_nn: torch.Tensor,
            color_to_play: chess.Color
    ) -> BoardEvaluation:
        ...

    @abstractmethod
    def to_nn_outputs(
            self,
            board_evaluation: BoardEvaluation
    ) -> torch.Tensor:
        ...


class OneDToValueWhite(OutputValueConverter):
    """
    Converting from a NN that output_converters a !D value from the point of view of the player to move
    """

    def convert_value_for_mover_viewpoint_to_value_white(
            self,
            turn: chess.Color,
            value_from_mover_view_point: float
    ) -> float:
        if turn == chess.BLACK:
            value_white = -value_from_mover_view_point
        else:
            value_white = value_from_mover_view_point
        return value_white

    def to_board_evaluation(
            self,
            output_nn: torch.Tensor,
            color_to_play: chess.Color
    ) -> BoardEvaluation:
        value: float = output_nn.item()
        value_white: float = self.convert_value_for_mover_viewpoint_to_value_white(
            turn=color_to_play,
            value_from_mover_view_point=value)
        board_evaluation: BoardEvaluation = BoardEvaluation(value_white=value_white)
        return board_evaluation

    def to_nn_outputs(
            self,
            board_evaluation: BoardEvaluation
    ) -> torch.Tensor:
        raise Exception('Not implemented in output_value_converter.py')
