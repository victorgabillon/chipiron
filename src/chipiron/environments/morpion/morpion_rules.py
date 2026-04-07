"""Morpion rules adapter."""

from dataclasses import dataclass

from valanga import SOLO

from chipiron.environments.morpion.types import MorpionState
from chipiron.games.domain.game.game_rules import (
    GameOutcome,
    GameRules,
    OutcomeKind,
    PositionAssessment,
)


@dataclass(frozen=True)
class MorpionRules(GameRules[MorpionState]):
    """Chipiron-facing rules adapter for Morpion Solitaire."""

    def outcome(self, state: MorpionState) -> GameOutcome | None:
        """Return game outcome for terminal Morpion states."""
        if state.is_game_over():
            return GameOutcome(
                kind=OutcomeKind.WIN,
                winner=SOLO,
                reason="no_legal_moves",
            )
        return None

    def pretty_result(self, state: MorpionState, outcome: GameOutcome) -> str:
        """Return a human-friendly game result string."""
        del state
        result = "completed" if outcome.kind is OutcomeKind.WIN else outcome.kind.value
        return f"Result: {result}" + (f" ({outcome.reason})" if outcome.reason else "")

    def assessment(  # pylint: disable=useless-return
        self, state: MorpionState
    ) -> PositionAssessment | None:
        """Return a non-terminal assessment for current state."""
        del state
        return None

    def pretty_assessment(
        self,
        state: MorpionState,
        assessment: PositionAssessment,
    ) -> str:
        """Return a human-friendly assessment string."""
        del state
        return f"Assessment: {assessment.kind.value}" + (
            f" ({assessment.reason})" if assessment.reason else ""
        )
