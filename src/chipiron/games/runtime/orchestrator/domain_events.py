"""Domain events emitted during match orchestration."""

from dataclasses import dataclass

import valanga
from valanga import BranchKey

from chipiron.displays.gui_protocol import Scope


@dataclass(frozen=True, slots=True)
class StartMatch:
    """Signal that a match has started with its initial state."""

    scope: Scope
    initial_state: valanga.TurnState


@dataclass(frozen=True, slots=True)
class NeedAction:
    """Request an action from the side to play for a specific state."""

    scope: Scope
    color: valanga.Color
    request_id: int
    state: valanga.TurnState


@dataclass(frozen=True, slots=True)
class ProposeAction:
    """Propose a candidate action for validation and application."""

    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey


@dataclass(frozen=True, slots=True)
class ActionApplied:
    """Report that an action was accepted and applied to the match."""

    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey


@dataclass(frozen=True, slots=True)
class IllegalAction:
    """Report that a proposed action was rejected as illegal."""

    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey | str
    reason: str


@dataclass(frozen=True, slots=True)
class MatchOver:
    """Signal that the match has reached a terminal state."""

    scope: Scope
    final_state: valanga.TurnState
    over_event: valanga.OverEvent | None


type MatchEvent = (
    StartMatch | NeedAction | ProposeAction | ActionApplied | IllegalAction | MatchOver
)
