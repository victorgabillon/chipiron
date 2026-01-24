"""
Chess-specific wiring for board evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import parsley_coco
from atomheart.board import IBoard
from coral.neural_networks.neural_net_board_eval_args import NeuralNetBoardEvalArgs

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    AllBoardEvaluatorArgs,
    BasicEvaluationBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.basic_evaluation import BasicEvaluation
from chipiron.players.boardevaluators.board_evaluator import StateEvaluator
from chipiron.players.boardevaluators.neural_networks.chipiron_nn_args import (
    create_content_to_input_from_model_weights,
)
from chipiron.players.boardevaluators.neural_networks.factory import (
    create_nn_board_eval_from_nn_parameters_file_and_existing_model,
)
from chipiron.players.boardevaluators.stockfish_board_evaluator import (
    StockfishBoardEvalArgs,
    StockfishBoardEvaluator,
)
from chipiron.utils.logger import chipiron_logger


@dataclass(frozen=True, slots=True)
class BoardEvalArgsWrapper:
    board_evaluator: AllBoardEvaluatorArgs


def _load_chipiron_eval_args() -> AllBoardEvaluatorArgs:
    path = str(
        files("chipiron").joinpath(
            "data/players/board_evaluator_config/base_chipiron_board_eval.yaml"
        )
    )
    wrapper: BoardEvalArgsWrapper = parsley_coco.resolve_yaml_file_to_base_dataclass(
        yaml_path=path,
        base_cls=BoardEvalArgsWrapper,
        package_name=str(files("chipiron")),
    )
    return wrapper.board_evaluator


class ValangaBoardEvaluator(StateEvaluator[ChessState]):
    def __init__(self, evaluator: StateEvaluator[IBoard]) -> None:
        self._evaluator = evaluator

    def value_white(self, state: ChessState) -> float:
        return self._evaluator.value_white(state.board)


def _build_chi() -> StateEvaluator[ChessState]:
    args = _load_chipiron_eval_args()
    match args:
        case BasicEvaluationBoardEvaluatorArgs():
            return ValangaBoardEvaluator(BasicEvaluation())
        case NeuralNetBoardEvalArgs():
            nn = args.neural_nets_model_and_architecture
            return ValangaBoardEvaluator(
                create_nn_board_eval_from_nn_parameters_file_and_existing_model(
                    model_weights_file_name=nn.model_weights_file_name,
                    nn_architecture_args=nn.nn_architecture_args,
                    content_to_input_convert=create_content_to_input_from_model_weights(
                        nn.model_weights_file_name
                    ),
                )
            )
        case _:
            raise ValueError(f"Unsupported chi args: {args!r}")


def _build_oracle(*, can_oracle: bool) -> StateEvaluator[ChessState] | None:
    if not can_oracle:
        return None
    if not StockfishBoardEvaluator.is_available():
        chipiron_logger.warning(
            "Stockfish binary not available; disabling oracle evaluator."
        )
        return None
    return ValangaBoardEvaluator(
        StockfishBoardEvaluator(StockfishBoardEvalArgs(depth=20, time_limit=0.1))
    )


@dataclass(frozen=True, slots=True)
class ChessEvalWiring:
    can_oracle: bool

    def build_chi(self) -> StateEvaluator[ChessState]:
        return _build_chi()

    def build_oracle(self) -> StateEvaluator[ChessState] | None:
        return _build_oracle(can_oracle=self.can_oracle)
