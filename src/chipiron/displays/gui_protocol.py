"""Protocol payloads exchanged between the GUI and game runtime."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Never

from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import Color, StateTag
from valanga.evaluations import Value

from chipiron.core.request_context import RequestContext
from chipiron.core.roles import GameRole, ParticipantId
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_playing_status import PlayingStatus

type SchemaVersion = int
type SessionId = str
type MatchId = str
type GameId = str


class GuiProtocolError(AssertionError):
    """Base error for unexpected GUI protocol values."""


class UnhandledGuiProtocolValueError(GuiProtocolError):
    """Raised when an unexpected value is received in the GUI protocol."""

    def __init__(self, value: object) -> None:
        """Initialize the error with the unexpected value."""
        super().__init__(f"Unhandled value: {value!r}")


@dataclass(frozen=True, slots=True)
class Scope:
    """Identify the session/match/game context for a GUI message."""

    session_id: SessionId
    match_id: MatchId | None
    game_id: GameId


def make_scope(
    *, session_id: SessionId, match_id: MatchId | None, game_id: GameId
) -> Scope:
    """Construct a scope from explicit identifiers."""
    return Scope(session_id=session_id, match_id=match_id, game_id=game_id)


def scope_for_new_game(existing_scope: Scope, new_game_id: GameId) -> Scope:
    """Create a scope for a new game while preserving session and match."""
    return Scope(
        session_id=existing_scope.session_id,
        match_id=existing_scope.match_id,
        game_id=new_game_id,
    )


def assert_never(x: Never) -> Never:
    """Fail fast for unreachable enum/union branches."""
    raise UnhandledGuiProtocolValueError(x)


# ---------- Updates (game -> gui) ----------
@dataclass(frozen=True, slots=True)
class UpdStateChess:
    """Snapshot update for a chess position."""

    state_tag: StateTag
    fen_plus_history: FenPlusHistory
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class UpdStateGeneric:
    """Game-agnostic snapshot update for rendering and history display."""

    state_tag: StateTag
    action_name_history: Sequence[str]
    adapter_payload: Any
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class UpdParticipantProgress:
    """Progress update for a specific game role."""

    role: GameRole
    progress_percent: int | None

    @property
    def player_color(self) -> GameRole:
        """Backward-compatible alias while some callers still say `player_color`."""
        return self.role


@dataclass(frozen=True, slots=True)
class UpdEvaluation:
    """Evaluation update payload."""

    oracle: Value | None
    chipiron: Value | None
    white: Value | None = None
    black: Value | None = None


@dataclass(frozen=True, slots=True)
class ParticipantUiInfo:
    """User-facing label and control hint for one role assignment."""

    role: GameRole
    role_label: str
    label: str
    is_human: bool


@dataclass(frozen=True, slots=True)
class UpdParticipantsInfo:
    """Update payload describing every role-participant binding for the game."""

    participants: Sequence[ParticipantUiInfo]

    def participant_by_role(self, role: GameRole) -> ParticipantUiInfo:
        """Return the participant entry for a specific role."""
        for participant in self.participants:
            if participant.role == role:
                return participant
        raise KeyError(role)

    @property
    def white(self) -> ParticipantUiInfo:
        """Backward-compatible accessor for legacy white/black games."""
        return self.participant_by_role(Color.WHITE)

    @property
    def black(self) -> ParticipantUiInfo:
        """Backward-compatible accessor for legacy white/black games."""
        return self.participant_by_role(Color.BLACK)


@dataclass(frozen=True, slots=True)
class ParticipantMatchStats:
    """Aggregate result counts for one participant in the current match."""

    participant_id: ParticipantId
    wins: int
    losses: int
    draws: int
    unknown: int


@dataclass(frozen=True, slots=True)
class UpdMatchResults:
    """Aggregate match results payload."""

    participant_stats: Sequence[ParticipantMatchStats]
    draws: int
    games_played: int
    match_finished: bool

    @property
    def wins_white(self) -> int:
        """Backward-compatible score accessor for the first participant."""
        if not self.participant_stats:
            return 0
        return self.participant_stats[0].wins

    @property
    def wins_black(self) -> int:
        """Backward-compatible score accessor for the second participant."""
        if len(self.participant_stats) < 2:
            return 0
        return self.participant_stats[1].wins


@dataclass(frozen=True, slots=True)
class UpdNeedHumanAction:
    """Update payload telling GUI which human request is currently pending."""

    ctx: RequestContext
    state_tag: StateTag


@dataclass(frozen=True, slots=True)
class UpdNoHumanActionPending:
    """Update payload telling GUI there is no pending human action."""


@dataclass(frozen=True, slots=True)
class UpdGameStatus:
    """Current game status update."""

    status: PlayingStatus


type UpdatePayload = (
    UpdStateChess
    | UpdStateGeneric
    | UpdParticipantProgress
    | UpdEvaluation
    | UpdParticipantsInfo
    | UpdMatchResults
    | UpdGameStatus
    | UpdNeedHumanAction
    | UpdNoHumanActionPending
)

PlayerUiInfo = ParticipantUiInfo
UpdPlayersInfo = UpdParticipantsInfo
UpdPlayerProgress = UpdParticipantProgress


@dataclass(frozen=True, slots=True)
class GuiUpdate:
    """GUI update envelope with schema and scope metadata."""

    schema_version: SchemaVersion
    game_kind: GameKind
    scope: Scope
    payload: UpdatePayload


# ---------- Commands (gui -> game) ----------
@dataclass(frozen=True, slots=True)
class CmdBackOneMove:
    """Command to rewind a single move in the current game."""


@dataclass(frozen=True, slots=True)
class CmdSetStatus:
    """Command to set the game playing status."""

    status: PlayingStatus


@dataclass(frozen=True, slots=True)
class HumanActionChosen:
    """Command carrying a generic human action name."""

    action_name: str
    ctx: RequestContext | None = None
    corresponding_state_tag: StateTag | None = None


type CommandPayload = CmdBackOneMove | CmdSetStatus | HumanActionChosen


@dataclass(frozen=True, slots=True)
class GuiCommand:
    """GUI command envelope with schema and scope metadata."""

    schema_version: SchemaVersion
    scope: Scope
    payload: CommandPayload
