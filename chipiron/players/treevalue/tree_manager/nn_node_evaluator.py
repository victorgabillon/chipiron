import torch
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import BoardEvaluation
from .node_evaluator import NodeEvaluator
from typing import List
import chipiron.players.treevalue.nodes as nodes
from chipiron.players.boardevaluators.neural_networks.board_evaluator import NNBoardEvaluator


class NNNodeEvaluator(NodeEvaluator):
    """ The Generic Neural network class for board evaluation"""

    def __init__(self,
                 nn_board_evaluator: NNBoardEvaluator,
                 syzygy: object
                 ) -> None:
        super().__init__(
            board_evaluator=nn_board_evaluator,
            syzygy=syzygy)
        self.net = nn_board_evaluator.net
        self.my_scripted_model = torch.jit.script(self.net)

    def evaluate_all_not_over(
            self,
            not_over_nodes: List[nodes.AlgorithmNode]
    ):
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
            board_evaluation: BoardEvaluation = self.board_evaluator.output_and_value_converter.to_board_evaluation(
                output_nn=output_layer[index],
                color_to_play=node_not_over.tree_node.board.turn)
            value_white: float = board_evaluation.value_white
            processed_evaluation = self.process_evalution_not_over(value_white, node_not_over)
            node_not_over.minmax_evaluation.set_evaluation(processed_evaluation)
