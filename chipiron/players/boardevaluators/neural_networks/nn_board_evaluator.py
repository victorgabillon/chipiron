"""
Module for the Neural Network Board Evaluator
"""
import chess
import torch

import chipiron.environments.chess.board as boards
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import BoardToInput
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import \
    OutputValueConverter
from chipiron.utils.chi_nn import ChiNN


class NNBoardEvaluator:
    """
    The Generic Neural network class for board evaluation

    Attributes:
        net (ChiNN): The neural network model
        output_and_value_converter (OutputValueConverter): The converter for output values
        board_to_input_converter (BoardToInput): The converter for board to input tensor
    """

    net: ChiNN
    output_and_value_converter: OutputValueConverter
    board_to_input_converter: BoardToInput

    def __init__(
            self,
            net: ChiNN,
            output_and_value_converter: OutputValueConverter,
            board_to_input_converter: BoardToInput
    ) -> None:
        """
        Initialize the NNBoardEvaluator

        Args:
            net (ChiNN): The neural network model
            output_and_value_converter (OutputValueConverter): The converter for output values
            board_to_input_converter (BoardToInput): The converter for board to input tensor
        """
        self.net = net
        self.my_scripted_model = torch.jit.script(net)
        self.output_and_value_converter = output_and_value_converter
        self.board_to_input_converter = board_to_input_converter

    def value_white(
            self,
            board: boards.BoardChi
    ) -> float:
        """
        Evaluate the value for the white player

        Args:
            board (BoardChi): The chess board

        Returns:
            float: The value for the white player
        """
        self.my_scripted_model.eval()
        input_layer: torch.Tensor = self.board_to_input_converter.convert(board=board)
        torch.no_grad()
        output_layer: torch.Tensor = self.my_scripted_model(input_layer)
        torch.no_grad()
        board_evaluation: BoardEvaluation = self.output_and_value_converter.to_board_evaluation(
            output_nn=output_layer,
            color_to_play=board.turn)
        value_white: float = board_evaluation.value_white
        return value_white

    def evaluate(
            self,
            input_layer: torch.Tensor,
            color_to_play: chess.Color
    ) -> BoardEvaluation:
        """
        Evaluate the board position

        Args:
            input_layer (torch.Tensor): The input tensor representing the board position
            color_to_play (chess.Color): The color to play

        Returns:
            BoardEvaluation: The evaluation of the board position
        """
        self.my_scripted_model.eval()
        torch.no_grad()

        # run the batch of input_converters into the NN and get the batch of output_converters
        output_layer = self.my_scripted_model(input_layer)

        # translate the NN output batch into a proper Board Evaluations classes in a list
        board_evaluations = self.output_and_value_converter.to_board_evaluation(
            output_nn=output_layer,
            color_to_play=color_to_play
        )

        return board_evaluations
