import torch
import chess
from src.players.boardevaluators.board_evaluator import BoardEvaluator


class NNBoardEval(BoardEvaluator):
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
        print('$$%%', input_layers, output_layer, len(not_over_nodes))
        for index, node_not_over in enumerate(not_over_nodes):
            predicted_value_from_mover_view_point = output_layer[index].item()
            # print('%%%##~', predicted_value_from_mover_view_point)
            value_white_eval = self.convert_value_for_mover_viewpoint_to_value_white(node_not_over,
                                                                                     predicted_value_from_mover_view_point)
            processed_evaluation = self.process_evalution_not_over(value_white_eval, node_not_over)
            node_not_over.set_evaluation(processed_evaluation)
