"""
Fallback evaluator wiring for games without a specific evaluator.
"""

from dataclasses import dataclass

from chipiron.players.boardevaluators.board_evaluator import StateEvaluator


class ConstantEvaluator:
    """Evaluator that always returns a fixed value."""

    def __init__(self, value: float = 0.0) -> None:
        """Store the constant value to return."""
        self._value = value

    def value_white(self, state: object) -> float:
        """Value white."""
        _ = state
        return self._value


@dataclass(frozen=True, slots=True)
class NullEvalWiring:
    """Wiring that provides the constant evaluator and no oracle."""

    def build_chi(self) -> StateEvaluator[object]:
        """Build the constant evaluator."""
        return ConstantEvaluator()

    def build_oracle(self) -> StateEvaluator[object] | None:
        """Build oracle."""
        return None
