"""Evaluator wiring for Morpion."""

from dataclasses import dataclass

from chipiron.environments.morpion.types import MorpionState
from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

from .morpion_state_evaluator import MorpionStateEvaluator


@dataclass(frozen=True, slots=True)
class MorpionEvalWiring:
    """Provide the simple Morpion heuristic and no oracle evaluator."""

    def build_chi(self) -> StateEvaluator[MorpionState]:
        """Build the primary Morpion evaluator."""
        return MorpionStateEvaluator()

    def build_oracle(self) -> StateEvaluator[MorpionState] | None:
        """Morpion currently has no exact oracle evaluator."""
        return None
