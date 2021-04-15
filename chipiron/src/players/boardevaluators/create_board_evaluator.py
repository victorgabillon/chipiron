import sys
from src.players.boardevaluators.basic_evaluation import BasicEvaluation
from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1

from src.players.boardevaluators.syzygy import Syzygy
from src.players.boardevaluators.board_evaluators_wrapper import BoardEvaluatorsWrapper


def create_board_evaluator(arg, syzygy):
    assert (isinstance(syzygy, Syzygy))

    print('----dedfr', arg)
    if arg['type'] == 'basic_evaluation':
        board_evaluator = BasicEvaluation()
    elif arg['type'] == 'nn_pp1':
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        board_evaluator = NetPP1('', arg['nn_param_file_name'])
        board_evaluator.load_or_init_weights()
    else:
        sys.exit('cant find ' + arg['type'])

    if arg['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    return BoardEvaluatorsWrapper(board_evaluator, syzygy_)
