"""Integer reduction rules adapter."""

from dataclasses import dataclass

from valanga import SOLO

from chipiron.environments.integer_reduction.types import IntegerReductionState
from chipiron.games.domain.game.game_rules import (
    GameOutcome,
    GameRules,
    OutcomeKind,
    PositionAssessment,
)


@dataclass(frozen=True)
class IntegerReductionRules(GameRules[IntegerReductionState]):
    """Chipiron-facing rules adapter for integer reduction.

    This is a true solo-game result mapping: reaching ``1`` is a win for the
    single ``SOLO`` role, with no synthetic opponent or loser role invented by
    the reporting layer.
    """

    def outcome(self, state: IntegerReductionState) -> GameOutcome | None:
        """Return game outcome for terminal integer-reduction states."""
        if state.is_game_over():
            return GameOutcome(
                kind=OutcomeKind.WIN,
                winner=SOLO,
                reason="reached_one",
            )
        return None

    def pretty_result(
        self, state: IntegerReductionState, outcome: GameOutcome
    ) -> str:
        """Return a human-friendly game result string."""
        del state
        result = "success" if outcome.kind is OutcomeKind.WIN else outcome.kind.value
        return f"Result: {result}" + (f" ({outcome.reason})" if outcome.reason else "")

    def assessment(  # pylint: disable=useless-return
        self, state: IntegerReductionState
    ) -> PositionAssessment | None:
        """Return a non-terminal assessment for current state."""
        del state
        return None

    def pretty_assessment(
        self,
        state: IntegerReductionState,
        assessment: PositionAssessment,
    ) -> str:
        """Return a human-friendly assessment string."""
        del state
        return f"Assessment: {assessment.kind.value}" + (
            f" ({assessment.reason})" if assessment.reason else ""
        )
