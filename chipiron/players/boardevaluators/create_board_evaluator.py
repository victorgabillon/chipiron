import sys
from players.boardevaluators.basic_evaluation import BasicEvaluation
from players.boardevaluators.neural_networks.nn_pp1 import NetPP1

from players.boardevaluators.syzygy import Syzygy
from players.boardevaluators.board_evaluators_wrapper import BoardEvaluatorsWrapper


def create_board_evaluator(arg, syzygy):
    assert (isinstance(syzygy, Syzygy))

    print('----dedfr', arg)
    if arg['type'] == 'BasicEvaluation':
        board_evaluator = BasicEvaluation()
    elif arg['type'] == 'nn_pp1':
        board_evaluator = NetPP1('', arg['nn_param_file_name'])
    else:
        sys.exit('cant find ' + arg['type'])

    if arg['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    return BoardEvaluatorsWrapper(board_evaluator, syzygy_)
