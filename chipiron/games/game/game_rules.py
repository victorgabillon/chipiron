"""
Protocol and data structures for game-specific rules and outcomes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

from valanga import Color

from .final_game_result import FinalGameResult

StateT = TypeVar("StateT",contravariant=True)


class OutcomeKind(str, Enum):
    WIN = "win"
    DRAW = "draw"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class OutcomeSource(str, Enum):
    TERMINAL = "terminal"
    ADJUDICATED = "adjudicated"


class VerdictKind(str, Enum):
    WIN = "win"
    DRAW = "draw"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GameOutcome:
    kind: OutcomeKind
    winner: Color | None = None
    reason: str | None = None
    source: OutcomeSource = OutcomeSource.TERMINAL


@dataclass(frozen=True)
class PositionAssessment:
    kind: VerdictKind
    winner: Color | None = None
    reason: str | None = None


class GameRules(Protocol[StateT]):
    def outcome(self, state: StateT) -> GameOutcome | None:
        """Return None if not terminal; otherwise a GameOutcome."""
        ...

    def pretty_result(self, state: StateT, outcome: GameOutcome) -> str:
        """Return a human-friendly string describing the outcome."""
        ...

    def assessment(self, state: StateT) -> PositionAssessment | None:
        """Return a non-terminal assessment such as tablebase verdicts."""
        ...

    def pretty_assessment(
        self, state: StateT, assessment: PositionAssessment
    ) -> str:
        """Return a human-friendly string describing the assessment."""
        ...


def outcome_to_final_game_result(outcome: GameOutcome) -> FinalGameResult:
    """
    Map a generic game outcome to the legacy FinalGameResult enum.

    Unknown/aborted outcomes map to draws to preserve legacy behavior when
    a game stops before a clean adjudication is available.
    """
    if outcome.kind is OutcomeKind.WIN:
        if outcome.winner is Color.WHITE:
            return FinalGameResult.WIN_FOR_WHITE
        if outcome.winner is Color.BLACK:
            return FinalGameResult.WIN_FOR_BLACK
        raise ValueError("Winner must be provided for WIN outcomes.")
    if outcome.kind is OutcomeKind.DRAW:
        return FinalGameResult.DRAW
    if outcome.kind in (OutcomeKind.ABORTED, OutcomeKind.UNKNOWN):
        return FinalGameResult.DRAW
    raise ValueError(f"Unhandled outcome kind: {outcome.kind}")
