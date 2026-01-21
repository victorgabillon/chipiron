"""
Module for building game state evaluators (stockfish + chipiron) with optional GUI publishing.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import TypeVar

import parsley_coco

from valanga import StateEvaluation

from chipiron.players.boardevaluators.all_board_evaluator_args import (
    AllBoardEvaluatorArgs,
    BasicEvaluationBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.stockfish_board_evaluator import (
    StockfishBoardEvalArgs,
    StockfishBoardEvaluator,
)
from coral.neural_networks.neural_net_board_eval_args import NeuralNetBoardEvalArgs

from .board_evaluator import (
    GameStateEvaluator,
    IGameStateEvaluator,
    ObservableGameStateEvaluator,
    StateEvaluator,
)
from .neural_networks.factory import (
    create_nn_board_eval_from_nn_parameters_file_and_existing_model,
)

StateT = TypeVar("StateT")

from typing import Protocol, TypeVar

StateT = TypeVar("StateT")

class EvaluatorWiring(Protocol[StateT]):
    def build_chi(self) -> StateEvaluator[StateT]: ...
    def build_oracle(self) -> StateEvaluator[StateT] | None: ...


# ---------------- IO (config loading) ----------------

@dataclass(frozen=True, slots=True)
class BoardEvalArgsWrapper:
    board_evaluator: AllBoardEvaluatorArgs


def load_chipiron_eval_args() -> AllBoardEvaluatorArgs:
    chi_board_eval_yaml_path = str(
        files("chipiron").joinpath(
            "data/players/board_evaluator_config/base_chipiron_board_eval.yaml"
        )
    )

    wrapper: BoardEvalArgsWrapper = parsley_coco.resolve_yaml_file_to_base_dataclass(
        yaml_path=chi_board_eval_yaml_path,
        base_cls=BoardEvalArgsWrapper,
        package_name=str(files("chipiron")),
    )
    return wrapper.board_evaluator


# ---------------- Pure builders ----------------

def build_state_evaluator[StateT](args: AllBoardEvaluatorArgs) -> StateEvaluator[StateT]:
    match args:
        case StockfishBoardEvalArgs():
            return StockfishBoardEvaluator(args)
        case BasicEvaluationBoardEvaluatorArgs():
            return BasicEvaluation()
        case NeuralNetBoardEvalArgs():
            nn = args.neural_nets_model_and_architecture
            return create_nn_board_eval_from_nn_parameters_file_and_existing_model(
                model_weights_file_name=nn.model_weights_file_name,
                nn_architecture_args=nn.nn_architecture_args,
            )
        case _:
            raise ValueError(f"Unsupported evaluator args: {args!r}")


def build_game_state_evaluator[StateT](
    *,
    chi: StateEvaluator[StateT],
    stock: StateEvaluator[StateT] | None,
) -> GameStateEvaluator[StateT]:
    return GameStateEvaluator(
        board_evaluator_stock=stock,
        board_evaluator_chi=chi,
    )


# ---------------- Public entrypoint ----------------

def make_game_state_evaluator[StateT](
    *,
    gui: bool,
    can_stockfish: bool,
    chi_args: AllBoardEvaluatorArgs | None = None,
    stockfish_args: StockfishBoardEvalArgs = StockfishBoardEvalArgs(depth=20, time_limit=0.1),
) -> IGameStateEvaluator[StateT]:
    if chi_args is None:
        chi_args = load_chipiron_eval_args()

    chi = build_state_evaluator[StateT](chi_args)

    stock: StateEvaluator[StateT] | None = None
    if can_stockfish:
        stock = build_state_evaluator[StateT](stockfish_args)

    base: IGameStateEvaluator[StateT] = build_game_state_evaluator(chi=chi, stock=stock)

    if gui:
        return ObservableGameStateEvaluator(base)

    return base
