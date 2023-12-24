import sys
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval, NeuralNetBoardEvalArgs, \
    NNBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from players.move_selector.treevalue.node_evaluator import NodeEvaluator
from .nn_node_evaluator import NNNodeEvaluator
from dataclasses import dataclass


@dataclass
class NeuralNetNodeEvalArgs:
    type: str
    neural_network: NeuralNetBoardEvalArgs
    representation: str
    syzygy_evaluation: bool

    def __post_init__(self):
        if self.type != 'neural_network':
            raise ValueError('Expecting neural_network as name')


NodeEvaluatorsArgs = NeuralNetNodeEvalArgs


def create_node_evaluator(
        arg_board_evaluator: NodeEvaluatorsArgs,
        syzygy: SyzygyTable
) -> NodeEvaluator:
    if arg_board_evaluator.syzygy_evaluation:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    node_evaluator: NodeEvaluator
    match arg_board_evaluator:
        case 'basic_evaluation':
            board_evaluator: BasicEvaluation = BasicEvaluation()
            node_evaluator: NodeEvaluator = NodeEvaluator(
                board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case NeuralNetNodeEvalArgs():
            board_evaluator: NNBoardEvaluator = create_nn_board_eval(arg=arg_board_evaluator.neural_network)
            node_evaluator: NodeEvaluator = NNNodeEvaluator(
                nn_board_evaluator=board_evaluator,
                syzygy=syzygy_
            )
        case other:
            sys.exit(f'Board Eval: can not find {other}')

    return node_evaluator
