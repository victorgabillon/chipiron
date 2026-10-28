"""Module for oracles."""

from collections.abc import Hashable
from typing import Any, Protocol

from valanga import State

type BranchName = str
type Value = Any


class PolicyOracle[StateT_contra: State](Protocol):
    """Generic oracle for game-specific best-action queries."""

    def supports(self, state: StateT_contra) -> bool:
        """Return whether this oracle can answer for the given state."""
        ...

    def recommend(self, state: StateT_contra) -> BranchName:
        """Return the recommended branch for the given state."""
        ...


class ValueOracle[StateT_contra: State](Protocol):
    """Generic oracle for game-specific exact evaluation."""

    def supports(self, state: StateT_contra) -> bool:
        """Return whether this oracle can evaluate the given state."""
        ...

    def evaluate(self, state: StateT_contra) -> Value:
        """Return the exact value for the given state."""
        ...


class TerminalOracle[StateT_contra: State, RoleT_co: Hashable](Protocol):
    """Oracle for providing terminal metadata (winner/draw)."""

    def supports(self, state: StateT_contra) -> bool:
        """Return whether terminal metadata is available for the state."""
        ...

    def over_event(self, state: StateT_contra) -> Any:
        """Return the terminal event associated with the given state."""
        ...
