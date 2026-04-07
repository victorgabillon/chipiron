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

if TYPE_CHECKING:
    from collections.abc import Hashable

    from chipiron.core.evaluation_scale import EvaluationScale
    from chipiron.environments.integer_reduction.types import IntegerReductionState


def _terminal_over_event() -> OverEvent:
    """Build the canonical solo terminal over-event for reaching one."""
    return OverEvent(
        outcome=Outcome.WIN,
        termination="reached_one",
        winner=None,
    )


@dataclass(frozen=True, slots=True)
class IntegerReductionStateEvaluator:
    """Step-aware evaluator aligned with the environment reward ``-steps``."""

    def score(self, state: IntegerReductionState) -> float:
        """Score a state so maximizing search prefers shorter paths."""
        return -float(state.steps)

    def evaluate(self, state: IntegerReductionState) -> Value:
        """Evaluate a state for Chipiron's generic GUI/game-state evaluator."""
        if state.is_game_over():
            return Value(
                score=self.score(state),
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

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return a terminal over-event and value when the goal state is reached."""
        integer_state = cast("IntegerReductionState", state)
        if not integer_state.is_game_over():
            return None, None
        return _terminal_over_event(), -float(integer_state.steps)


@dataclass(frozen=True, slots=True)
class IntegerReductionMasterEvaluator(MasterStateValueEvaluator):
    """Anemone-compatible integer-reduction evaluator for tree search."""

    evaluator: IntegerReductionStateEvaluator
    over: OverEventDetector
    over_detector: IntegerReductionOverEventDetector

    def evaluate(self, state: State) -> Value:
        """Evaluate a state so tree search maximizes shorter successful paths."""
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
    del evaluation_scale
    evaluator = IntegerReductionStateEvaluator()
    over_detector = IntegerReductionOverEventDetector()
    return IntegerReductionMasterEvaluator(
        evaluator=evaluator,
        over=over_detector,
        over_detector=over_detector,
    )
