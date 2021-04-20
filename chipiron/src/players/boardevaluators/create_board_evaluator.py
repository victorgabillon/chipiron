import sys
from src.players.boardevaluators.basic_evaluation import BasicEvaluation
from src.players.boardevaluators.neural_networks.nn_board_eval import NNBoardEval
from src.players.boardevaluators.syzygy import Syzygy
from src.players.boardevaluators.board_evaluators_wrapper import BoardEvaluatorsWrapper


def create_board_evaluator(arg, syzygy):
    assert (isinstance(syzygy, Syzygy))

    if arg['type'] == 'basic_evaluation':
        board_evaluator = BasicEvaluation()
    elif arg['type'] == 'neural_network':
        board_evaluator = NNBoardEval(arg)
    else:
        sys.exit('Board Eval: can not find ' + arg['type'])

    if arg['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    return BoardEvaluatorsWrapper(board_evaluator, syzygy_)
