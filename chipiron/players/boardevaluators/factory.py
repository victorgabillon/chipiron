from chipiron.players.boardevaluators.stockfish_board_evaluator import StockfishBoardEvaluator
from .board_evaluator import BoardEvaluator, ObservableBoardEvaluator, BoardEvaluatorWrapped
import sys
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval
import yaml


def create_board_evaluator(
        args_board_evaluator: dict,
) -> BoardEvaluator:
    board_evaluator: BoardEvaluator
    match args_board_evaluator['type']:
        case 'stockfish':
            board_evaluator = StockfishBoardEvaluator()
        case 'basic_evaluation':
            board_evaluator = BasicEvaluation()
        case 'neural_network':
            board_evaluator = create_nn_board_eval(args_board_evaluator['neural_network'])
        case 'table_base':
            board_evaluator = None
        case other:
            sys.exit(f'Board Eval: cannot find {other}')

    return board_evaluator


def create_board_evaluator_wrapped(
        args_board_evaluator: dict,
        syzygy: object
) -> BoardEvaluatorWrapped:
    board_evaluator: BoardEvaluator = create_board_evaluator(args_board_evaluator)

    if args_board_evaluator['syzygy_evaluation']:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    board_evaluator_wrapped: BoardEvaluatorWrapped = BoardEvaluatorWrapped(
        board_evaluator=node_evaluator,
        syzygy=syzygy_
    )

    return board_evaluator


class ObservableBoardEvaluatorFactory:
    def __init__(self):
        self.subscribers = []

    def create(self):
        board_evaluator_stock: BoardEvaluator = create_board_evaluator(
            args_board_evaluator={'type': 'stockfish'}
        )

        chi_board_eval_yaml_path: str = 'data/players/board_evaluator_config/base_chipiron_board_eval.yaml'
        with open(chi_board_eval_yaml_path, 'r') as chi_board_eval_yaml_file:
            chi_board_eval_dict: dict = yaml.load(chi_board_eval_yaml_file, Loader=yaml.FullLoader)
            board_evaluator_chi: BoardEvaluator = create_board_evaluator(
                args_board_evaluator=chi_board_eval_dict
            )

        board_evaluator_table: BoardEvaluator = create_board_evaluator(
            args_board_evaluator={'type': 'table_base'}
        )

        if self.subscribers:

            board_evaluator: ObservableBoardEvaluator = ObservableBoardEvaluator(
                board_evaluator_stock=board_evaluator_stock,
                board_evaluator_chi=board_evaluator_chi
            )
            for subscriber in self.subscribers:
                board_evaluator.subscribe(subscriber)
        return board_evaluator

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
