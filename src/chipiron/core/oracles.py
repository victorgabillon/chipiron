"""Module for oracles."""

from typing import Protocol, TypeVar

from valanga import State
from valanga.game import BranchName
from valanga.over_event import OverEvent

StateT_contra = TypeVar("StateT_contra", bound=State, contravariant=True)


class PolicyOracle(Protocol[StateT_contra]):
    """Generic oracle for game-specific best-action queries."""

    def supports(self, state: StateT_contra) -> bool:
        """Whether the oracle can recommend a move for this state."""
        ...

    def recommend(self, state: StateT_contra) -> BranchName:
        """Return the recommended branch name for the given state."""
        ...


class ValueOracle(Protocol[StateT_contra]):
    """Generic oracle for game-specific exact evaluation."""

    def supports(self, state: StateT_contra) -> bool:
        """Whether the oracle can evaluate this state."""
        ...

    def value_white(self, state: StateT_contra) -> float:
        """Return the evaluation from White's perspective."""
        ...


class TerminalOracle(Protocol[StateT_contra]):
    """Oracle for providing terminal metadata (winner/draw)."""

    def supports(self, state: StateT_contra) -> bool:
        """Whether the oracle can report a terminal outcome for this state."""
        ...

    def over_event(self, state: StateT_contra) -> OverEvent:
        """Return a terminal OverEvent for the given state."""
        ...
