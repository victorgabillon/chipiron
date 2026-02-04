"""Protocol payloads exchanged between the GUI and game runtime."""

from dataclasses import dataclass
from typing import Never, TypeAlias

from atomheart.board.utils import FenPlusHistory
from valanga import Color
from valanga.evaluations import StateEvaluation

from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus

SchemaVersion: TypeAlias = int
SessionId: TypeAlias = str
MatchId: TypeAlias = str
GameId: TypeAlias = str


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
    raise AssertionError(f"Unhandled value: {x!r}")


# ---------- Updates (game -> gui) ----------
@dataclass(frozen=True, slots=True)
class UpdStateChess:
    """Snapshot update for a chess position."""

    fen_plus_history: FenPlusHistory
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class UpdPlayerProgress:
    """Progress update for a specific player."""

    player_color: Color
    progress_percent: int | None


@dataclass(frozen=True, slots=True)
class UpdEvaluation:
    """Evaluation update payload."""

    oracle: StateEvaluation | None
    chipiron: StateEvaluation | None
    white: StateEvaluation | None = None
    black: StateEvaluation | None = None


@dataclass(frozen=True, slots=True)
class PlayerUiInfo:
    """User-facing label and control hint for a player."""

    label: str
    is_human: bool


@dataclass(frozen=True, slots=True)
class UpdPlayersInfo:
    """Update payload describing both players."""

    white: PlayerUiInfo
    black: PlayerUiInfo


@dataclass(frozen=True, slots=True)
class UpdMatchResults:
    """Aggregate match results payload."""

    wins_white: int
    wins_black: int
    draws: int
    games_played: int
    match_finished: bool


@dataclass(frozen=True, slots=True)
class UpdGameStatus:
    """Current game status update."""

    status: PlayingStatus


UpdatePayload: TypeAlias = (
    UpdStateChess
    | UpdPlayerProgress
    | UpdEvaluation
    | UpdPlayersInfo
    | UpdMatchResults
    | UpdGameStatus
)


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

    pass


@dataclass(frozen=True, slots=True)
class CmdSetStatus:
    """Command to set the game playing status."""

    status: PlayingStatus


@dataclass(frozen=True, slots=True)
class CmdHumanMoveUci:
    """Command carrying a human UCI move and optional context."""

    move_uci: str
    corresponding_fen: str | None = None
    color_to_play: Color | None = None


CommandPayload: TypeAlias = CmdBackOneMove | CmdSetStatus | CmdHumanMoveUci


@dataclass(frozen=True, slots=True)
class GuiCommand:
    """GUI command envelope with schema and scope metadata."""

    schema_version: SchemaVersion
    scope: Scope
    payload: CommandPayload
