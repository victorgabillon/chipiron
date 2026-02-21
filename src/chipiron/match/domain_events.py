from dataclasses import dataclass

import valanga
from valanga.dynamics import Transition
from valanga import BranchKey

from chipiron.displays.gui_protocol import Scope


@dataclass(frozen=True, slots=True)
class StartMatch:
    scope: Scope
    initial_state: valanga.TurnState


@dataclass(frozen=True, slots=True)
class NeedAction:
    scope: Scope
    color: valanga.Color
    request_id: int
    state: valanga.TurnState


@dataclass(frozen=True, slots=True)
class ProposeAction:
    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey


@dataclass(frozen=True, slots=True)
class ActionApplied:
    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey
    transition: Transition[valanga.TurnState]


@dataclass(frozen=True, slots=True)
class IllegalAction:
    scope: Scope
    color: valanga.Color
    request_id: int
    action: BranchKey | str
    reason: str


@dataclass(frozen=True, slots=True)
class MatchOver:
    scope: Scope
    final_state: valanga.TurnState
    over_event: valanga.OverEvent | None
