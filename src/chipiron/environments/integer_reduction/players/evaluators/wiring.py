"""Evaluator wiring for integer reduction."""

from dataclasses import dataclass

from chipiron.environments.integer_reduction.types import IntegerReductionState
from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

from .integer_reduction_state_evaluator import IntegerReductionStateEvaluator


@dataclass(frozen=True, slots=True)
class IntegerReductionEvalWiring:
    """Provide the simple integer-reduction heuristic and no oracle evaluator."""

    def build_chi(self) -> StateEvaluator[IntegerReductionState]:
        """Build the primary integer-reduction evaluator."""
        return IntegerReductionStateEvaluator()

    def build_oracle(self) -> StateEvaluator[IntegerReductionState] | None:
        """Integer reduction currently has no exact oracle evaluator."""
        return None
