"""Simple Morpion evaluators used by GUI scoring and tree search."""

# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from valanga import Outcome, State
from valanga.evaluations import Certainty, Value
from valanga.over_event import OverEvent

from chipiron.environments.morpion.types import MorpionDynamics

if TYPE_CHECKING:
    from chipiron.core.evaluation_scale import EvaluationScale
    from chipiron.environments.morpion.types import MorpionState


class MorpionTermination(StrEnum):
    """Termination reasons for Morpion terminal over-events."""

    NO_LEGAL_MOVES = "no_legal_moves"


def _terminal_over_event() -> OverEvent[Any]:
    """Build the canonical solo terminal over-event for Morpion completion."""
    return OverEvent[Any](
        outcome=Outcome.WIN,
        termination=MorpionTermination.NO_LEGAL_MOVES,
        winner=None,
    )


@dataclass(frozen=True, slots=True)
class MorpionStateEvaluator:
    """Minimal Morpion evaluator aligned with single-agent maximizing search."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)

    def legal_action_count(self, state: MorpionState) -> int:
        """Return the current number of legal Morpion moves."""
        return self.dynamics.legal_action_count(state)

    def _score_from_legal_action_count(
        self,
        *,
        moves: int,
        legal_action_count: int,
    ) -> float:
        """Project one Morpion position to the minimal scalar heuristic used in v1."""
        return float(moves) + 0.01 * float(legal_action_count)

    def score(self, state: MorpionState) -> float:
        """Score a state so maximizing search prefers deeper, still-open runs."""
        return self._score_from_legal_action_count(
            moves=state.moves,
            legal_action_count=self.legal_action_count(state),
        )

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
            score=self._score_from_legal_action_count(
                moves=state.moves,
                legal_action_count=legal_action_count,
            ),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


@dataclass(frozen=True, slots=True)
class MorpionOverEventDetector:
    """Detect terminal solo completions for Morpion search."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent[Any] | None, float | None]:
        """Return a terminal over-event and value when no legal move remains."""
        morpion_state = cast("MorpionState", state)
        legal_action_count = self.dynamics.legal_action_count(morpion_state)
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
                over_event=over_event,
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
