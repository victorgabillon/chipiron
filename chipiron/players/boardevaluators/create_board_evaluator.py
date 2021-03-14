import sys
from players.boardevaluators.basic_evaluation import BasicEvaluation
#from players.boardevaluators.NN1 import NN1
from players.boardevaluators.NN1_pytorch import NN1Pytorch
from players.boardevaluators.NN2_pytorch import NN2Pytorch
from players.boardevaluators.NN4_pytorch import NN4Pytorch

from players.boardevaluators.syzygy import Syzygy
from players.boardevaluators.board_evaluators_wrapper import BoardEvaluatorsWrapper


def create_board_evaluator(arg, syzygy):
    assert (isinstance(syzygy, Syzygy))

    print('----dedfr', arg)
    if arg['type'] == 'BasicEvaluation':
        board_evaluator = BasicEvaluation()
   # elif arg['type'] == 'NN1':
   #     board_evaluator = NN1(arg['nn_name'])
    elif arg['type'] == 'NN1Pytorch':
        board_evaluator = NN1Pytorch(arg['nn_param_file_name'])
    elif arg['type'] == 'NN2Pytorch':
        board_evaluator = NN2Pytorch(arg['nn_param_file_name'])
    elif arg['type'] == 'NN4Pytorch':
        board_evaluator = NN4Pytorch('',arg['nn_param_file_name'])
    else:
        sys.exit('cant find ' + arg['type'])

    if arg['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    return BoardEvaluatorsWrapper(board_evaluator, syzygy_)
