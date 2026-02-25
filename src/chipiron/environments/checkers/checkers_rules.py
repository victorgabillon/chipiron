"""Checkers-specific rules adapter.

Atomheart's CheckersRules is a pure configuration dataclass; terminal detection
lives in dynamics + move generation. Chipiron's GameRules protocol expects
outcome/assessment without stepping the game, so we reproduce the same terminal
logic used by atomheart.CheckersDynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomheart.games.checkers.generation import generate_legal_moves
from atomheart.games.checkers.state import CheckersState
from valanga import Color

from chipiron.games.game.game_rules import (
    GameOutcome,
    GameRules,
    OutcomeKind,
    PositionAssessment,
)

if TYPE_CHECKING:
    from atomheart.games.checkers.rules import CheckersRules as AtomCheckersRules


@dataclass(frozen=True)
class CheckersRules(GameRules[CheckersState]):
    """Chipiron-facing rules adapter for checkers."""

    inner: AtomCheckersRules

    def outcome(self, state: CheckersState) -> GameOutcome | None:
        """Return game outcome for terminal states."""
        if state.is_game_over():
            winner = Color.BLACK if (state.wm | state.wk) == 0 else Color.WHITE
            return GameOutcome(
                kind=OutcomeKind.WIN,
                winner=winner,
                reason="piece_exhaustion",
            )

        legal_moves = generate_legal_moves(state=state, rules=self.inner)
        if len(legal_moves) == 0:
            winner = Color.BLACK if state.turn == Color.WHITE else Color.WHITE
            return GameOutcome(kind=OutcomeKind.WIN, winner=winner, reason="no_moves")

        return None

    def pretty_result(self, state: CheckersState, outcome: GameOutcome) -> str:
        """Return a human-friendly game result string."""
        del state
        if outcome.kind is OutcomeKind.WIN:
            result = "1-0" if outcome.winner is Color.WHITE else "0-1"
        elif outcome.kind is OutcomeKind.DRAW:
            result = "1/2-1/2"
        elif outcome.kind is OutcomeKind.ABORTED:
            result = "*"
        else:
            result = "?"

        return f"Result: {result}" + (f" ({outcome.reason})" if outcome.reason else "")

    def assessment(self, state: CheckersState) -> PositionAssessment | None:
        """Return a non-terminal assessment for current state."""
        del state
        assessment: PositionAssessment | None = None
        return assessment

    def pretty_assessment(
        self,
        state: CheckersState,
        assessment: PositionAssessment,
    ) -> str:
        """Return a human-friendly assessment string."""
        del state
        return f"Assessment: {assessment.kind.value}" + (
            f" ({assessment.reason})" if assessment.reason else ""
        )
