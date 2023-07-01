import torch
import chess
from chipiron.players.boardevaluators.node_evaluator import NodeEvaluator
from chipiron.extra_tools.chi_nn import ChiNN
from chipiron.players.boardevaluators.neural_networks.output_value_converter import OutputValueConverter
from typing import List
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation
import chipiron.players.treevalue.nodes as nodes


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

    def value_white(self, node: nodes.AlgorithmNode):
        # self.net.eval()
        self.my_scripted_model.eval()
        input_layer = node.board_representation.get_evaluator_input(color_to_play=node.player_to_move)
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

    def evaluate_all_not_over(self, not_over_nodes: List[nodes.AlgorithmNode]):
        list_of_tensors = [0] * len(not_over_nodes)
        index: int
        node_not_over: nodes.AlgorithmNode
        for index, node_not_over in enumerate(not_over_nodes):
            list_of_tensors[index] = node_not_over.board_representation.get_evaluator_input(
                color_to_play=node_not_over.player_to_move
            )
        input_layers = torch.stack(list_of_tensors, dim=0)
        self.my_scripted_model.eval()
        torch.no_grad()

        output_layer = self.my_scripted_model(input_layers)

        for index, node_not_over in enumerate(not_over_nodes):
            predicted_value_from_mover_view_point = output_layer[index].item()
            value_white_eval = self.convert_value_for_mover_viewpoint_to_value_white(node_not_over.tree_node,
                                                                                     predicted_value_from_mover_view_point)
            processed_evaluation = self.process_evalution_not_over(value_white_eval, node_not_over)
            node_not_over.minmax_evaluation.set_evaluation(processed_evaluation)
