"""Models used by the customtkinter script GUI."""

from dataclasses import dataclass, field
from enum import StrEnum

from chipiron.environments.types import GameKind

from .participant_selection import ParticipantSelection
from .registries import launcher_spec_for_game


def _empty_participants() -> list[ParticipantSelection]:
    """Return an empty typed participant list for dataclass defaults."""
    return []


class ScriptGUIType(StrEnum):
    """The type of script to run based on GUI selection."""

    PLAY_OR_WATCH_A_GAME = "play_or_watch_a_game"
    TREE_VISUALIZATION = "tree_visualization"


@dataclass
class ArgsChosenByUser:
    """Arguments selected by the user in the GUI."""

    type: ScriptGUIType = ScriptGUIType.PLAY_OR_WATCH_A_GAME
    game_kind: GameKind = GameKind.CHESS
    starting_position_key: str = ""
    participants: list[ParticipantSelection] = field(default_factory=_empty_participants)

    def __post_init__(self) -> None:
        """Fill launcher defaults from the selected game's launcher spec."""
        if self.participants and self.starting_position_key:
            return

        launcher_spec = launcher_spec_for_game(self.game_kind)
        if not self.participants:
            self.participants = list(launcher_spec.default_participants)
        if not self.starting_position_key:
            self.starting_position_key = launcher_spec.default_starting_position_key
