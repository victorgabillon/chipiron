"""Models used by the customtkinter script GUI."""

from dataclasses import dataclass
from enum import StrEnum

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag


class ScriptGUIType(StrEnum):
    """The type of script to run based on GUI selection."""

    PLAY_OR_WATCH_A_GAME = "play_or_watch_a_game"
    TREE_VISUALIZATION = "tree_visualization"


@dataclass
class ArgsChosenByUser:
    """Arguments selected by the user in the GUI."""

    type: ScriptGUIType = ScriptGUIType.PLAY_OR_WATCH_A_GAME
    game_kind: GameKind = GameKind.CHESS
    player_type_white: PlayerConfigTag = PlayerConfigTag.RECUR_ZIPF_BASE_3
    strength_white: int | None = 1
    player_type_black: PlayerConfigTag = PlayerConfigTag.RECUR_ZIPF_BASE_3
    strength_black: int | None = 1
    starting_position_key: str = "Standard"
