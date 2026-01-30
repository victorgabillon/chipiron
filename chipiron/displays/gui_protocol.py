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
    session_id: SessionId
    match_id: MatchId | None
    game_id: GameId


def make_scope(
    *, session_id: SessionId, match_id: MatchId | None, game_id: GameId
) -> Scope:
    return Scope(session_id=session_id, match_id=match_id, game_id=game_id)


def scope_for_new_game(existing_scope: Scope, new_game_id: GameId) -> Scope:
    return Scope(
        session_id=existing_scope.session_id,
        match_id=existing_scope.match_id,
        game_id=new_game_id,
    )


def assert_never(x: Never) -> Never:
    raise AssertionError(f"Unhandled value: {x!r}")


# ---------- Updates (game -> gui) ----------
@dataclass(frozen=True, slots=True)
class UpdStateChess:
    fen_plus_history: FenPlusHistory
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class UpdPlayerProgress:
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
    label: str
    is_human: bool


@dataclass(frozen=True, slots=True)
class UpdPlayersInfo:
    white: PlayerUiInfo
    black: PlayerUiInfo


@dataclass(frozen=True, slots=True)
class UpdMatchResults:
    wins_white: int
    wins_black: int
    draws: int
    games_played: int
    match_finished: bool


@dataclass(frozen=True, slots=True)
class UpdGameStatus:
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
    schema_version: SchemaVersion
    game_kind: GameKind
    scope: Scope
    payload: UpdatePayload


# ---------- Commands (gui -> game) ----------
@dataclass(frozen=True, slots=True)
class CmdBackOneMove:
    pass


@dataclass(frozen=True, slots=True)
class CmdSetStatus:
    status: PlayingStatus


@dataclass(frozen=True, slots=True)
class CmdHumanMoveUci:
    move_uci: str
    corresponding_fen: str | None = None
    color_to_play: Color | None = None


CommandPayload: TypeAlias = CmdBackOneMove | CmdSetStatus | CmdHumanMoveUci


@dataclass(frozen=True, slots=True)
class GuiCommand:
    schema_version: SchemaVersion
    scope: Scope
    payload: CommandPayload
