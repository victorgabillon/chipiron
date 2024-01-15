import sys
import chipiron.players.boardevaluators.basic_evaluation as basic_evaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval, NNBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.move_selector.treevalue.node_evaluator import NodeEvaluator
from chipiron.players.move_selector.treevalue.node_evaluator.neural_networks.nn_node_evaluator import NNNodeEvaluator
from .all_node_evaluators import NodeEvaluatorTypes
from chipiron.players.move_selector.treevalue.node_evaluator import NodeEvaluatorArgs

from . import neural_networks

AllNodeEvaluatorArgs = neural_networks.NeuralNetNodeEvalArgs | NodeEvaluatorArgs


def create_node_evaluator(
        arg_board_evaluator: AllNodeEvaluatorArgs,
        syzygy: SyzygyTable
) -> NodeEvaluator:
    if arg_board_evaluator.syzygy_evaluation:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    node_evaluator: NodeEvaluator

    match arg_board_evaluator.type:
        case 'basic_evaluation':
            board_evaluator: basic_evaluation.BasicEvaluation = basic_evaluation.BasicEvaluation()
            node_evaluator: NodeEvaluator = NodeEvaluator(
                board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case NodeEvaluatorTypes.NeuralNetwork:
            board_evaluator: NNBoardEvaluator = create_nn_board_eval(arg=arg_board_evaluator.neural_network)
            node_evaluator: NodeEvaluator = NNNodeEvaluator(
                nn_board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case other:
            sys.exit(f'Node Board Eval: can not find {other} in file {__name__}')

    return node_evaluator
