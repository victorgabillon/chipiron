"""Generic args for tree-and-value master state evaluators."""

from dataclasses import dataclass

from chipiron.core.evaluation_scale import EvaluationScale
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    AllBoardEvaluatorArgs,
)


@dataclass
class MasterBoardEvaluatorArgs:
    """Shared evaluator configuration consumed by player pipeline wiring."""

    board_evaluator: AllBoardEvaluatorArgs
    oracle_evaluation: bool = False
    evaluation_scale: EvaluationScale = EvaluationScale.ENTIRE_REAL_AXIS
