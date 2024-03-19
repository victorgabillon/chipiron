import sys
from typing import TypeAlias

import chipiron.players.boardevaluators.basic_evaluation as basic_evaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import NNBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from . import neural_networks
from .all_node_evaluators import NodeEvaluatorTypes
from .neural_networks.nn_node_evaluator import NNNodeEvaluator
from .node_evaluator import NodeEvaluator
from .node_evaluator_args import NodeEvaluatorArgs

AllNodeEvaluatorArgs: TypeAlias = neural_networks.NeuralNetNodeEvalArgs | NodeEvaluatorArgs


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
            node_evaluator = NodeEvaluator(
                board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case NodeEvaluatorTypes.NeuralNetwork:
            assert isinstance(arg_board_evaluator, neural_networks.NeuralNetNodeEvalArgs)
            board_evaluator_nn: NNBoardEvaluator = create_nn_board_eval(arg=arg_board_evaluator.neural_network)
            node_evaluator = NNNodeEvaluator(
                nn_board_evaluator=board_evaluator_nn,
                syzygy=syzygy_
            )
        case other:
            sys.exit(f'Node Board Eval: can not find {other} in file {__name__}')

    return node_evaluator
