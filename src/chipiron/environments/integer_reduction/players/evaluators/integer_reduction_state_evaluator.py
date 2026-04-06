"""Simple integer-reduction evaluators used by GUI scoring and tree search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from valanga import Outcome, State
from valanga.evaluations import Certainty, Value
from valanga.over_event import OverEvent

from chipiron.core.evaluation_scale import EvaluationScale

if TYPE_CHECKING:
    from collections.abc import Hashable

    from chipiron.environments.integer_reduction.types import IntegerReductionState

TERMINAL_SCORE_ENTIRE_REAL_AXIS = 1_000_000.0
TERMINAL_SCORE_UNIT_INTERVAL = 1.0


def _terminal_score_for_scale(evaluation_scale: EvaluationScale) -> float:
    """Return a terminal reward consistent with the configured evaluation scale."""
    if evaluation_scale is EvaluationScale.SYMMETRIC_UNIT_INTERVAL:
        return TERMINAL_SCORE_UNIT_INTERVAL
    return TERMINAL_SCORE_ENTIRE_REAL_AXIS


def _terminal_over_event() -> OverEvent:
    """Build the canonical solo terminal over-event for reaching one."""
    return OverEvent(
        outcome=Outcome.WIN,
        termination="reached_one",
        winner=None,
    )


@dataclass(frozen=True, slots=True)
class IntegerReductionStateEvaluator:
    """Heuristic evaluator where smaller integers are better for the solo player."""

    terminal_score: float = TERMINAL_SCORE_ENTIRE_REAL_AXIS

    def score(self, state: IntegerReductionState) -> float:
        """Score a state so maximizing search prefers smaller values."""
        if state.is_game_over():
            return self.terminal_score
        return -float(state.value)

    def evaluate(self, state: IntegerReductionState) -> Value:
        """Evaluate a state for Chipiron's generic GUI/game-state evaluator."""
        if state.is_game_over():
            return Value(
                score=self.terminal_score,
                certainty=Certainty.TERMINAL,
                over_event=_terminal_over_event(),
            )

        return Value(
            score=self.score(state),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


@dataclass(frozen=True, slots=True)
class IntegerReductionOverEventDetector:
    """Detect terminal solo wins for integer reduction search."""

    terminal_score: float = TERMINAL_SCORE_ENTIRE_REAL_AXIS

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return a terminal over-event and value when the goal state is reached."""
        integer_state = cast("IntegerReductionState", state)
        if not integer_state.is_game_over():
            return None, None
        return _terminal_over_event(), self.terminal_score


@dataclass(frozen=True, slots=True)
class IntegerReductionMasterEvaluator(MasterStateValueEvaluator):
    """Anemone-compatible integer-reduction evaluator for tree search."""

    evaluator: IntegerReductionStateEvaluator
    over: OverEventDetector
    over_detector: IntegerReductionOverEventDetector

    def evaluate(self, state: State) -> Value:
        """Evaluate a state so tree search maximizes progress toward one."""
        over_event, terminal_value = self.over_detector.check_obvious_over_events(state)
        if terminal_value is not None:
            return Value(
                score=terminal_value,
                certainty=Certainty.TERMINAL,
                over_event=cast("OverEvent[Hashable] | None", over_event),
            )

        return Value(
            score=self.evaluator.score(cast("IntegerReductionState", state)),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


def build_integer_reduction_master_evaluator(
    *,
    evaluation_scale: EvaluationScale,
) -> IntegerReductionMasterEvaluator:
    """Build the integer-reduction master evaluator for tree-based selectors."""
    terminal_score = _terminal_score_for_scale(evaluation_scale)
    evaluator = IntegerReductionStateEvaluator(terminal_score=terminal_score)
    over_detector = IntegerReductionOverEventDetector(terminal_score=terminal_score)
    return IntegerReductionMasterEvaluator(
        evaluator=evaluator,
        over=over_detector,
        over_detector=over_detector,
    )
