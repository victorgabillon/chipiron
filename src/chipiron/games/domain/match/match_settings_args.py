"""Module to define the MatchSettingsArgs dataclass."""

from dataclasses import dataclass

from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_tag import GameConfigTag
from chipiron.games.domain.match.match_role_schedule import (
    SoloMatchSchedule,
    TwoRoleMatchSchedule,
)


@dataclass
class MatchSettingsArgs:
    """Dataclass to store match settings arguments.

    Match scheduling is expressed explicitly by topology-aware neutral schedule
    objects rather than white/black-specific quota fields.

    """

    schedule: SoloMatchSchedule | TwoRoleMatchSchedule
    game_args: GameConfigTag | GameArgs
