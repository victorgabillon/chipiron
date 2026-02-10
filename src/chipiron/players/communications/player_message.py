"""Message payloads exchanged between player processes and the manager."""

from dataclasses import dataclass

from valanga import Color, StateEvaluation, StateTag
from valanga.game import BranchName, Seed

from chipiron.displays.gui_protocol import Scope


@dataclass(frozen=True, slots=True)
class TurnStatePlusHistory[StateSnapT = object]:
    """Picklable state container used in player <-> manager communications.

    `StateSnapT` must be picklable; for chess this should be a `FenPlusHistory`
    (which includes the python-chess private board stack when needed).
    """

    current_state_tag: StateTag
    turn: Color
    snapshot: StateSnapT
    historical_actions: list[str] | None = None


@dataclass(frozen=True, slots=True)
class PlayerRequest[StateSnapT = object]:
    """Manager -> player request.

    The request is game-agnostic: game-specific decoding happens inside the
    player's implementation (e.g., chess uses `FenPlusHistory`).
    """

    schema_version: int
    scope: Scope
    seed: Seed
    state: TurnStatePlusHistory[StateSnapT]


@dataclass(frozen=True, slots=True)
class EvMove:
    """Event reporting a chosen move and its evaluation."""

    branch_name: BranchName
    corresponding_state_tag: StateTag
    player_name: str
    color_to_play: Color
    evaluation: StateEvaluation | None = (
        None  # replace with your StateEvaluation type if needed
    )


@dataclass(frozen=True, slots=True)
class EvProgress:
    """Event reporting progress feedback for a player."""

    player_color: Color
    progress_percent: int | None


type PlayerEventPayload = EvMove | EvProgress


@dataclass(frozen=True, slots=True)
class PlayerEvent:
    """Envelope for player events sent to the manager."""

    schema_version: int
    scope: Scope
    payload: PlayerEventPayload
