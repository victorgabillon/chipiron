"""Simple Morpion evaluators used by GUI scoring and tree search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from valanga import Outcome, State
from valanga.evaluations import Certainty, Value
from valanga.over_event import OverEvent

from chipiron.environments.morpion.types import MorpionDynamics

if TYPE_CHECKING:
    from collections.abc import Hashable

    from chipiron.core.evaluation_scale import EvaluationScale
    from chipiron.environments.morpion.types import MorpionState


def _terminal_over_event() -> OverEvent:
    """Build the canonical solo terminal over-event for Morpion completion."""
    return OverEvent(
        outcome=Outcome.WIN,
        termination="no_legal_moves",
        winner=None,
    )


@dataclass(frozen=True, slots=True)
class MorpionStateEvaluator:
    """Minimal Morpion evaluator aligned with single-agent maximizing search."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)

    def legal_action_count(self, state: MorpionState) -> int:
        """Return the current number of legal Morpion moves."""
        return len(self.dynamics.legal_actions(state).get_all())

    def score(self, state: MorpionState) -> float:
        """Score a state so maximizing search prefers deeper, still-open runs."""
        legal_action_count = self.legal_action_count(state)
        return float(state.moves) + 0.01 * float(legal_action_count)

    def evaluate(self, state: MorpionState) -> Value:
        """Evaluate a state for Chipiron's generic GUI/game-state evaluator."""
        legal_action_count = self.legal_action_count(state)
        if legal_action_count == 0:
            return Value(
                score=float(state.moves),
                certainty=Certainty.TERMINAL,
                over_event=_terminal_over_event(),
            )

        return Value(
            score=float(state.moves) + 0.01 * float(legal_action_count),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


@dataclass(frozen=True, slots=True)
class MorpionOverEventDetector:
    """Detect terminal solo completions for Morpion search."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return a terminal over-event and value when no legal move remains."""
        morpion_state = cast("MorpionState", state)
        legal_action_count = len(self.dynamics.legal_actions(morpion_state).get_all())
        if legal_action_count != 0:
            return None, None
        return _terminal_over_event(), float(morpion_state.moves)


@dataclass(frozen=True, slots=True)
class MorpionMasterEvaluator(MasterStateValueEvaluator):
    """Anemone-compatible Morpion evaluator for tree search."""

    evaluator: MorpionStateEvaluator
    over: OverEventDetector
    over_detector: MorpionOverEventDetector

    def evaluate(self, state: State) -> Value:
        """Evaluate a state so tree search maximizes long productive runs."""
        over_event, terminal_value = self.over_detector.check_obvious_over_events(state)
        if terminal_value is not None:
            return Value(
                score=terminal_value,
                certainty=Certainty.TERMINAL,
                over_event=cast("OverEvent[Hashable] | None", over_event),
            )

        morpion_state = cast("MorpionState", state)
        return Value(
            score=self.evaluator.score(morpion_state),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


def build_morpion_master_evaluator(
    *,
    evaluation_scale: EvaluationScale,
) -> MorpionMasterEvaluator:
    """Build the Morpion master evaluator for tree-based selectors."""
    del evaluation_scale
    evaluator = MorpionStateEvaluator()
    over_detector = MorpionOverEventDetector()
    return MorpionMasterEvaluator(
        evaluator=evaluator,
        over=over_detector,
        over_detector=over_detector,
    )
