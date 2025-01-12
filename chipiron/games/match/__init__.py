"""
Module to manage matches.
"""

from .match_factories import create_match_manager
from .match_settings_args import MatchSettingsArgs

__all__ = ["MatchSettingsArgs", "create_match_manager"]
