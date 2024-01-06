import torch
import chess
from chipiron.utils.chi_nn import ChiNN
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import OutputValueConverter
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import BoardToInput
from typing import List
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation
import chipiron.environments.chess.board as boards


class NNBoardEvaluator:
    """ The Generic Neural network class for board evaluation"""

    net: ChiNN
    output_and_value_converter: OutputValueConverter
    board_to_input_converter: BoardToInput

    def __init__(self,
                 net: ChiNN,
                 output_and_value_converter: OutputValueConverter,
                 board_to_input_converter: BoardToInput
                 ) -> None:
        self.net = net
        self.my_scripted_model = torch.jit.script(net)
        self.output_and_value_converter = output_and_value_converter
        self.board_to_input_converter = board_to_input_converter

    def value_white(
            self,
            board: boards.BoardChi
    ) -> float:
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

    def evaluate(self, input_layer: torch.Tensor,
                 color_to_play: List[chess.Color]
                 ) -> List[BoardEvaluation]:
        self.my_scripted_model.eval()
        torch.no_grad()

        # run the batch of input_converters into the NN and get the batch of output_converters
        output_layer = self.my_scripted_model(input_layer)

        # translate the NN output batch into a proper Board Evaluations classes in a list
        board_evaluations = self.output_and_value_converter.to_board_evaluation(output_layer, color_to_play)

        return board_evaluations
