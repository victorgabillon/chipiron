from __future__ import annotations

from typing import Protocol, TypeVar

from valanga import State
from valanga.game import BranchName

StateT = TypeVar("StateT", bound=State)


class Oracle(Protocol[StateT]):
    """Generic oracle for game-specific best-action queries."""

    def supports(self, state: StateT) -> bool:
        """Whether the oracle can recommend a move for this state."""
        ...

    def recommend(self, state: StateT) -> BranchName:
        """Return the recommended branch name for the given state."""
        ...
