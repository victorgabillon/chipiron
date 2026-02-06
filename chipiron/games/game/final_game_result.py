"""Module for the FinalGameResult enum and the GameReport dataclass."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valanga import StateTag
    from valanga.game import ActionName
else:
    ActionName = str
    StateTag = str


class FinalGameResult(StrEnum):
    """Enum representing the final result of a game."""

    WIN_FOR_WHITE = "win_for_white"
    WIN_FOR_BLACK = "win_for_black"
    DRAW = "draw"


@dataclass
class GameReport:
    """Dataclass representing a game report."""

    final_game_result: FinalGameResult
    action_history: list[ActionName]
    state_tag_history: list[StateTag]
