"""
Fallback evaluator wiring for games without a specific evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

from chipiron.players.boardevaluators.board_evaluator import StateEvaluator


class ConstantEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def value_white(self, state: object) -> float:
        _ = state
        return self._value


@dataclass(frozen=True, slots=True)
class NullEvalWiring:
    def build_chi(self) -> StateEvaluator[object]:
        return ConstantEvaluator()

    def build_oracle(self) -> StateEvaluator[object] | None:
        return None
