"""Domain events emitted during match orchestration."""

from dataclasses import dataclass
from typing import Any

import valanga
from valanga import BranchKey

from chipiron.core.roles import GameRole
from chipiron.displays.gui_protocol import Scope

type AnyTurnState = valanga.TurnState[Any]


@dataclass(frozen=True, slots=True)
class StartMatch:
    """Signal that a match has started with its initial state."""

    scope: Scope
    initial_state: AnyTurnState


@dataclass(frozen=True, slots=True)
class NeedAction:
    """Request an action from the role to play for a specific state."""

    scope: Scope
    role: GameRole
    request_id: int
    state: AnyTurnState

    @property
    def color(self) -> GameRole:
        """Backward-compatible alias while runtime behavior stays color-shaped."""
        return self.role


@dataclass(frozen=True, slots=True)
class ProposeAction:
    """Propose a candidate action for validation and application."""

    scope: Scope
    role: GameRole
    request_id: int
    action: BranchKey

    @property
    def color(self) -> GameRole:
        """Backward-compatible alias while runtime behavior stays color-shaped."""
        return self.role


@dataclass(frozen=True, slots=True)
class ActionApplied:
    """Report that an action was accepted and applied to the match."""

    scope: Scope
    role: GameRole
    request_id: int
    action: BranchKey

    @property
    def color(self) -> GameRole:
        """Backward-compatible alias while runtime behavior stays color-shaped."""
        return self.role


@dataclass(frozen=True, slots=True)
class IllegalAction:
    """Report that a proposed action was rejected as illegal."""

    scope: Scope
    role: GameRole
    request_id: int
    action: BranchKey | str
    reason: str

    @property
    def color(self) -> GameRole:
        """Backward-compatible alias while runtime behavior stays color-shaped."""
        return self.role


@dataclass(frozen=True, slots=True)
class MatchOver:
    """Signal that the match has reached a terminal state."""

    scope: Scope
    final_state: AnyTurnState
    over_event: valanga.OverEvent[valanga.Role] | None


type MatchEvent = (
    StartMatch | NeedAction | ProposeAction | ActionApplied | IllegalAction | MatchOver
)
