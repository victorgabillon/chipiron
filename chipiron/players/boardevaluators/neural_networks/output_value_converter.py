from typing import List
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation, PointOfView
import torch
import chess
from abc import ABC, abstractmethod


class OutputValueConverter(ABC):
    """
    Converting an output of the neural network to a board evaluation
     and conversely convertinging a board evaluation to an output of the neural network
    """

    def __init__(self, point_of_view: PointOfView) -> None:
        self.point_of_view = point_of_view

    @abstractmethod
    def to_board_evaluation(self,
                            output_nn: torch.Tensor,
                            point_of_view: PointOfView,
                            color_to_play: List[chess.Color]) -> List[BoardEvaluation]:
        ...

    @abstractmethod
    def to_nn_outputs(self, board_evaluation: BoardEvaluation) -> torch.Tensor:
        ...


class OneDToValueWhite(OutputValueConverter):
    """
    Converting from a NN that outputs a !D value from the point of view of the player to move
    """

    def to_board_evaluation(self,
                            output_nn: torch.Tensor,
                            color_to_play: List[chess.Color]) -> List[BoardEvaluation]:

        board_evaluations = []
        for eval in output_nn:
            value_white = output_nn.item
            board_evaluation = BoardEvaluation(value_white=value_white)
            board_evaluations.append(board_evaluation)
        return board_evaluations

    def to_nn_outputs(self, board_evaluation: BoardEvaluation) -> torch.Tensor:
        ...
