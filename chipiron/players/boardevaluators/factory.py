from chipiron.players.boardevaluators.stockfish_board_evaluator import StockfishBoardEvaluator, StockfishBoardEvalArgs
from .board_evaluator import BoardEvaluator, ObservableBoardEvaluator, GameBoardEvaluator
import sys
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.neural_networks.factory import create_nn_board_eval, NeuralNetBoardEvalArgs
import yaml
import dacite
from dataclasses import dataclass
from typing import Any


class TableBaseArgs:
    ...


class BasicEvaluationArgs:
    ...


BoardEvalArgs = NeuralNetBoardEvalArgs | StockfishBoardEvalArgs | TableBaseArgs | BasicEvaluationArgs


@dataclass
class BoardEvalArgsWrapper:
    board_evaluator: BoardEvalArgs


def create_board_evaluator(
        args_board_evaluator: BoardEvalArgs,
) -> BoardEvaluator:
    board_evaluator: BoardEvaluator
    match args_board_evaluator:
        case StockfishBoardEvalArgs():
            board_evaluator = StockfishBoardEvaluator(args_board_evaluator)
        case BasicEvaluationArgs():
            board_evaluator = BasicEvaluation()
        case NeuralNetBoardEvalArgs():
            board_evaluator = create_nn_board_eval(args_board_evaluator)
        #  case TableBaseArgs():
        #      board_evaluator = None
        case other:
            sys.exit(f'Board Eval: cannot find {other} in file {__name__}')

    return board_evaluator


def create_game_board_evaluator_not_observable(
) -> GameBoardEvaluator:
    board_evaluator_stock: BoardEvaluator = create_board_evaluator(
        args_board_evaluator=StockfishBoardEvalArgs()
    )
    chi_board_eval_yaml_path: str = 'data/players/board_evaluator_config/base_chipiron_board_eval.yaml'
    with open(chi_board_eval_yaml_path, 'r') as chi_board_eval_yaml_file:
        chi_board_eval_dict: dict[Any, Any] = yaml.load(chi_board_eval_yaml_file, Loader=yaml.FullLoader)

        # atm using a wrapper because dacite does not accept unions as data_class argument
        chi_board_eval_args: BoardEvalArgsWrapper = dacite.from_dict(data_class=BoardEvalArgsWrapper,
                                                                     data=chi_board_eval_dict)
        board_evaluator_chi: BoardEvaluator = create_board_evaluator(
            args_board_evaluator=chi_board_eval_args.board_evaluator
        )

    # board_evaluator_table: BoardEvaluator = create_board_evaluator(
    #    args_board_evaluator=TableBaseArgs()
    # )

    game_board_evaluator: GameBoardEvaluator = GameBoardEvaluator(
        board_evaluator_stock=board_evaluator_stock,
        board_evaluator_chi=board_evaluator_chi
    )
    return game_board_evaluator


def create_game_board_evaluator(
        gui: bool
) -> ObservableBoardEvaluator | GameBoardEvaluator:
    game_board_evaluator: ObservableBoardEvaluator | GameBoardEvaluator
    game_board_evaluator = create_game_board_evaluator_not_observable()
    if gui:
        game_board_evaluator = ObservableBoardEvaluator(
            game_board_evaluator=game_board_evaluator
        )

    return game_board_evaluator
