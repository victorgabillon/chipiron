import torch
import chess
from src.players.boardevaluators.node_evaluator import NodeEvaluator, BoardEvaluator
from src.extra_tools.chi_nn import ChiNN
from src.players.boardevaluators.neural_networks.output_value_converter import OutputValueConverter
from typing import List
from src.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation, PointOfView


class NNBoardEvaluator:
    """ The Generic Neural network class for board evaluation"""

    def __init__(self, net: ChiNN,
                 output_and_value_converter: OutputValueConverter) -> None:
        self.net = net
        self.my_scripted_model = torch.jit.script(net)
        self.output_and_value_converter = output_and_value_converter

    def evaluate(self, input_layer: torch.Tensor,
                 color_to_play: List[chess.Color]
                 ) -> List[BoardEvaluation]:
        self.my_scripted_model.eval()
        torch.no_grad()

        # run the batch of inputs into the NN and get the batch of outputs
        output_layer = self.my_scripted_model(input_layer)

        # translate the NN output batch into a proper Board Evaluations classes in a list
        board_evaluations = self.output_and_value_converter.to_board_evaluation(output_layer, color_to_play)

        return board_evaluations


class NNNodeEvaluator(NodeEvaluator):
    """ The Generic Neural network class for board evaluation"""

    def __init__(self, net):
        self.net = net
        self.my_scripted_model = torch.jit.script(net)

    def compute_representation(self, node, parent_node, board_modifications):
        self.net.compute_representation(node, parent_node, board_modifications)

    def value_white(self, node):
        # self.net.eval()
        self.my_scripted_model.eval()
        input_layer = self.net.get_nn_input(node)
        torch.no_grad()
        output_layer = self.my_scripted_model(input_layer)
        torch.no_grad()
        predicted_value_from_mover_view_point = output_layer.item()
        value_white = self.convert_value_for_mover_viewpoint_to_value_white(node, predicted_value_from_mover_view_point)
        return value_white

    def convert_value_for_mover_viewpoint_to_value_white(self, node, value_from_mover_view_point):
        if node.board.turn == chess.BLACK:
            value_white = -value_from_mover_view_point
        else:
            value_white = value_from_mover_view_point
        return value_white

    def evaluate_all_not_over(self, not_over_nodes):
        list_of_tensors = [0] * len(not_over_nodes)
        for index, node_not_over in enumerate(not_over_nodes):
            list_of_tensors[index] = self.net.get_nn_input(node_not_over)
        input_layers = torch.stack(list_of_tensors, dim=0)
        self.my_scripted_model.eval()
        torch.no_grad()

        output_layer = self.my_scripted_model(input_layers)
        for index, node_not_over in enumerate(not_over_nodes):
            predicted_value_from_mover_view_point = output_layer[index].item()
            value_white_eval = self.convert_value_for_mover_viewpoint_to_value_white(node_not_over,
                                                                                     predicted_value_from_mover_view_point)
            processed_evaluation = self.process_evalution_not_over(value_white_eval, node_not_over)
            node_not_over.set_evaluation(processed_evaluation)
