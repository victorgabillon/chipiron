"""Module to manage matches."""

from .match_factories import create_match_manager
from .match_role_schedule import SoloMatchSchedule, TwoRoleMatchSchedule
from .match_settings_args import MatchSettingsArgs
from .match_tag import MatchConfigTag

__all__ = [
    "MatchConfigTag",
    "MatchSettingsArgs",
    "SoloMatchSchedule",
    "TwoRoleMatchSchedule",
    "create_match_manager",
]
