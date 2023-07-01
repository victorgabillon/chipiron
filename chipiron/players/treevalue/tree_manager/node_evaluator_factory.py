import sys
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from .node_evaluators_wrapper import NodeEvaluatorsWrapper

def create_node_evaluator(arg_board_evaluator: dict,
                          syzygy: SyzygyTable) -> NodeEvaluatorsWrapper:
    assert (isinstance(syzygy, SyzygyTable))

    match arg_board_evaluator['type'] :
        case 'basic_evaluation':
            board_evaluator = BasicEvaluation()
        case 'neural_network':
            board_evaluator = create_nn_board_eval(arg_board_evaluator['neural_network'])
        case other:
            sys.exit(f'Board Eval: can not find {other}')

    if arg_board_evaluator['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    node_evaluator: NodeEvaluatorsWrapper = NodeEvaluatorsWrapper(
        board_evaluator=board_evaluator,
        syzygy=syzygy_
    )

    return node_evaluator

