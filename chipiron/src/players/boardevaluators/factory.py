import sys
from src.players.boardevaluators.basic_evaluation import BasicEvaluation
from src.players.boardevaluators.neural_networks.factory import create_nn_board_eval
from src.players.boardevaluators.table_base.syzygy import SyzygyTable
from src.players.boardevaluators.board_evaluators_wrapper import BoardEvaluatorsWrapper


def create_board_evaluator(arg_board_evaluator, syzygy):
    assert (isinstance(syzygy, SyzygyTable))

    if arg_board_evaluator['type'] == 'basic_evaluation':
        board_evaluator = BasicEvaluation()
    elif arg_board_evaluator['type'] == 'neural_network':
        board_evaluator = create_nn_board_eval(arg_board_evaluator['neural_network'])
    else:
        sys.exit('Board Eval: can not find ' + arg_board_evaluator['type'])

    if arg_board_evaluator['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    return BoardEvaluatorsWrapper(board_evaluator, syzygy_)
