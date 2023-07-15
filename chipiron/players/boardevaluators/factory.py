from chipiron.players.boardevaluators.stockfish_board_evaluator import StockfishBoardEvaluator
from .board_evaluator import BoardEvaluator, ObservableBoardEvaluator, BoardEvaluatorWrapped, GameBoardEvaluator
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


def create_game_board_evaluator_not_observable() -> GameBoardEvaluator:
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

    game_board_evaluator: GameBoardEvaluator = GameBoardEvaluator(
        board_evaluator_stock=board_evaluator_stock,
        board_evaluator_chi=board_evaluator_chi
    )

    return game_board_evaluator


def create_game_board_evaluator(gui: bool):
    game_board_evaluator = create_game_board_evaluator_not_observable()
    if gui:
        game_board_evaluator = ObservableBoardEvaluator(
            game_board_evaluator=game_board_evaluator
        )

    return game_board_evaluator
