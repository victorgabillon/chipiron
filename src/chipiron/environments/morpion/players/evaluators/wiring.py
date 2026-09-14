"""Evaluator wiring for Morpion."""

from dataclasses import dataclass

from chipiron.environments.morpion.types import MorpionState
from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

from .morpion_state_evaluator import MorpionStateEvaluator
from .neural_evaluator_args import MorpionNeuralEvaluatorArgs


@dataclass(frozen=True, slots=True)
class MorpionEvalWiring:
    """Provide a configured neural evaluator or the default Morpion heuristic."""

    neural: MorpionNeuralEvaluatorArgs | None = None

    def build_chi(self) -> StateEvaluator[MorpionState]:
        """Build the primary Morpion evaluator."""
        if self.neural is not None:
            from .neural_evaluator import load_morpion_evaluator_from_model_bundle

            return load_morpion_evaluator_from_model_bundle(
                self.neural.model_bundle, device=self.neural.device
            )
        return MorpionStateEvaluator()

    def build_oracle(self) -> StateEvaluator[MorpionState] | None:
        """Morpion currently has no exact oracle evaluator."""
        return None
