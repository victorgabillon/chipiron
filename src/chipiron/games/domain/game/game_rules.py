"""Protocol and data structures for game-specific rules and outcomes."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from valanga import TurnState

from chipiron.core.roles import GameRole

from .final_game_result import RoleGameResult, RoleOutcome

type AnyTurnState = TurnState[Any]

StateT_contra = TypeVar("StateT_contra", contravariant=True, bound=AnyTurnState)


class OutcomeKind(StrEnum):
    """Kinds of game outcomes."""

    WIN = "win"
    DRAW = "draw"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class OutcomeSource(StrEnum):
    """Sources of game outcomes."""

    TERMINAL = "terminal"
    ADJUDICATED = "adjudicated"


class VerdictKind(StrEnum):
    """Kinds of position assessments."""

    WIN = "win"
    DRAW = "draw"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GameOutcome:
    """Represents the outcome of a game."""

    kind: OutcomeKind
    winner: GameRole | None = None
    reason: str | None = None
    source: OutcomeSource = OutcomeSource.TERMINAL


@dataclass(frozen=True)
class PositionAssessment:
    """Represents a non-terminal assessment of a position."""

    kind: VerdictKind
    winner: GameRole | None = None
    reason: str | None = None


class GameRules[StateT](Protocol):
    """Protocol for game rules."""

    def outcome(self, state: StateT) -> GameOutcome | None:
        """Return None if not terminal; otherwise a GameOutcome."""
        ...

    def pretty_result(self, state: StateT, outcome: GameOutcome) -> str:
        """Return a human-friendly string describing the outcome."""
        ...

    def assessment(self, state: StateT) -> PositionAssessment | None:
        """Return a non-terminal assessment such as tablebase verdicts."""
        ...

    def pretty_assessment(self, state: StateT, assessment: PositionAssessment) -> str:
        """Return a human-friendly string describing the assessment."""
        ...


class GameOutcomeError(ValueError):
    """Base error for invalid or unsupported game outcomes."""


class MissingWinnerError(GameOutcomeError):
    """Raised when a win outcome lacks a winner."""

    def __init__(self) -> None:
        """Initialize the error for a missing winner."""
        super().__init__("Winner must be provided for WIN outcomes.")


class UnhandledOutcomeKindError(GameOutcomeError):
    """Raised when an outcome kind is not handled."""

    def __init__(self, kind: OutcomeKind) -> None:
        """Initialize the error with the unhandled outcome kind."""
        super().__init__(f"Unhandled outcome kind: {kind}")


def outcome_to_role_game_result(
    outcome: GameOutcome, roles: tuple[GameRole, ...]
) -> RoleGameResult:
    """Map a generic game outcome to a role-aware result structure."""
    if outcome.kind is OutcomeKind.WIN:
        if outcome.winner is None:
            raise MissingWinnerError
        result_by_role = {
            role: RoleOutcome.WIN if role == outcome.winner else RoleOutcome.LOSS
            for role in roles
        }
        return RoleGameResult(
            result_by_role=result_by_role,
            winner_roles=(outcome.winner,),
            reason=outcome.reason,
        )
    if outcome.kind is OutcomeKind.DRAW:
        return RoleGameResult(
            result_by_role={role: RoleOutcome.DRAW for role in roles},
            reason=outcome.reason,
        )
    if outcome.kind in (OutcomeKind.ABORTED, OutcomeKind.UNKNOWN):
        return RoleGameResult(
            result_by_role={role: RoleOutcome.UNKNOWN for role in roles},
            reason=outcome.reason,
        )
    raise UnhandledOutcomeKindError(outcome.kind)
