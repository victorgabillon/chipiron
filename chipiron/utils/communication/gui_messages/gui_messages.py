"""
Module that contains the GUI messages that are sent to the GUI from the game.
"""

from dataclasses import dataclass

import chess

from chipiron.games.game.game_playing_status import PlayingStatus


from chipiron.environments.types import GameKind


from dataclasses import dataclass
from typing import   Literal, Mapping, Optional, TypeAlias, Union

from valanga import Color

from dataclasses import dataclass
from typing import   Literal, Mapping, Optional, TypeAlias, Union, Never

from atomheart.board.utils import FenPlusHistory

# --- tiny helper for exhaustiveness ---
def assert_never(x: Never) -> Never:
    raise AssertionError(f"Unhandled value: {x!r}")


# --- ids ---
GameId: TypeAlias = str
SchemaVersion: TypeAlias = int

# pyright: reportPrivateUsage=false  
BoardStateStack: TypeAlias = list[chess._BoardState]  # pyright: ignore[reportPrivateUsage]

# ---------- Updates (game -> gui) ----------
# ---------- Updates (game -> gui) ----------
@dataclass(frozen=True, slots=True)
class UpdStateChess:
    kind: Literal["state_chess"]
    fen_plus_history: FenPlusHistory
    seed: Optional[int] = None


@dataclass(frozen=True, slots=True)
class UpdPlayerProgress:
    kind: Literal["player_progress"]
    player_color: Color
    progress_percent: Optional[int]  # 0..100


@dataclass(frozen=True, slots=True)
class UpdEvaluation:
    kind: Literal["evaluation"]
    stock: Optional[float]
    chipiron: Optional[float]
    white: Optional[float] = None
    black: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PlayerInfo:
    name: str
    engine_kind: str
    extra: Mapping[str, str]  # keep it simple & stable


@dataclass(frozen=True, slots=True)
class UpdPlayersInfo:
    kind: Literal["players_info"]
    white: PlayerInfo
    black: PlayerInfo


@dataclass(frozen=True, slots=True)
class UpdMatchResults:
    kind: Literal["match_results"]
    wins_white: int
    wins_black: int
    draws: int
    games_played: int
    match_finished: bool


@dataclass(frozen=True, slots=True)
class UpdGameStatus:
    kind: Literal["game_status"]
    status: PlayingStatus


UpdatePayload: TypeAlias = Union[
    UpdStateChess,
    UpdPlayerProgress,
    UpdEvaluation,
    UpdPlayersInfo,
    UpdMatchResults,
    UpdGameStatus,
]


@dataclass(frozen=True, slots=True)
class GuiUpdate:
    schema_version: SchemaVersion
    game_kind: GameKind
    game_id: GameId
    payload: UpdatePayload


# ---------- Commands (gui -> game) ----------
@dataclass(frozen=True, slots=True)
class CmdBackOneMove:
    kind: Literal["back_one_move"]


@dataclass(frozen=True, slots=True)
class CmdSetStatus:
    kind: Literal["set_status"]
    status: PlayingStatus


@dataclass(frozen=True, slots=True)
class CmdHumanMoveUci:
    kind: Literal["human_move_uci"]
    move_uci: str
    # optional context for validation / debug:
    corresponding_fen: Optional[str] = None
    player_name: Optional[str] = None
    color_to_play: Optional[Color] = None


CommandPayload: TypeAlias = Union[
    CmdBackOneMove,
    CmdSetStatus,
    CmdHumanMoveUci,
]


@dataclass(frozen=True, slots=True)
class GuiCommand:
    schema_version: SchemaVersion
    game_id: GameId
    payload: CommandPayload

