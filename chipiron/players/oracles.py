from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from valanga import State
from valanga.game import BranchName

if TYPE_CHECKING:
    from valanga.over_event import OverEvent

StateT = TypeVar("StateT", bound=State)


class PolicyOracle(Protocol[StateT]):
    """Generic oracle for game-specific best-action queries."""

    def supports(self, state: StateT) -> bool:
        """Whether the oracle can recommend a move for this state."""
        ...

    def recommend(self, state: StateT) -> BranchName:
        """Return the recommended branch name for the given state."""
        ...


class ValueOracle(Protocol[StateT]):
    """Generic oracle for game-specific exact evaluation."""

    def supports(self, state: StateT) -> bool:
        """Whether the oracle can evaluate this state."""
        ...

    def value_white(self, state: StateT) -> float:
        """Return the evaluation from White's perspective."""
        ...


class TerminalOracle(Protocol[StateT]):
    """Oracle for providing terminal metadata (winner/draw)."""

    def supports(self, state: StateT) -> bool:
        """Whether the oracle can report a terminal outcome for this state."""
        ...

    def over_event(self, state: StateT) -> OverEvent:
        """Return a terminal OverEvent for the given state."""
        ...
