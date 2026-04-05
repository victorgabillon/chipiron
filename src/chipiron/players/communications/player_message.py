"""Message payloads exchanged between player processes and the manager."""

from dataclasses import dataclass

from valanga import StateTag
from valanga.evaluations import Value
from valanga.game import BranchName, Seed

from chipiron.core.request_context import RequestContext
from chipiron.core.roles import GameRole
from chipiron.displays.gui_protocol import Scope


@dataclass(frozen=True, slots=True)
class TurnStatePlusHistory[StateSnapT = object]:
    """Picklable state container used in player <-> manager communications.

    `StateSnapT` must be picklable; for chess this should be a `FenPlusHistory`
    (which includes the python-chess private board stack when needed).
    """

    current_state_tag: StateTag
    role_to_play: GameRole
    snapshot: StateSnapT
    historical_actions: list[str] | None = None

    @property
    def turn(self) -> GameRole:
        """Backward-compatible alias while callers still expect `turn`."""
        return self.role_to_play


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
    ctx: RequestContext | None = None


@dataclass(frozen=True, slots=True)
class EvMove:
    """Event reporting a chosen move and its evaluation."""

    branch_name: BranchName
    corresponding_state_tag: StateTag
    ctx: RequestContext | None
    player_name: str
    player_role: GameRole
    evaluation: Value | None = None  # replace with your Value type if needed

    @property
    def color_to_play(self) -> GameRole:
        """Backward-compatible alias while the runtime still uses colors."""
        return self.player_role


@dataclass(frozen=True, slots=True)
class EvProgress:
    """Event reporting progress feedback for a player."""

    player_role: GameRole
    progress_percent: int | None

    @property
    def player_color(self) -> GameRole:
        """Backward-compatible alias while GUI payloads remain color-shaped."""
        return self.player_role


type PlayerEventPayload = EvMove | EvProgress


@dataclass(frozen=True, slots=True)
class PlayerEvent:
    """Envelope for player events sent to the manager."""

    schema_version: int
    scope: Scope
    payload: PlayerEventPayload
