import sys
from src.players.boardevaluators.basic_evaluation import BasicEvaluation
from src.players.boardevaluators.neural_networks.factory import create_nn_board_eval
from src.players.boardevaluators.table_base.syzygy import SyzygyTable
from src.players.boardevaluators.board_evaluators_wrapper import NodeEvaluatorsWrapper
from src.players.boardevaluators.stockfish_board_evaluator import StockfishBoardEvaluator
from src.players.boardevaluators.board_evaluator import ObservableBoardEvaluator


def create_node_evaluator(arg_board_evaluator, syzygy):
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

    return NodeEvaluatorsWrapper(board_evaluator, syzygy_)


class BoardEvaluatorFactory:

    def create(self):
        board_evaluator = StockfishBoardEvaluator()
        return board_evaluator


class ObservableBoardEvaluatorFactory(BoardEvaluatorFactory):
    def __init__(self):
        self.subscribers = []

    def create(self):
        board_evaluator = super().create()
        if self.subscribers:

            board_evaluator = ObservableBoardEvaluator(board_evaluator)
            for subscriber in self.subscribers:
                board_evaluator.subscribe(subscriber)
        return board_evaluator

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
