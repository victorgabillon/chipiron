"""
This module defines the `MatchArgs` class, which represents the input arguments needed by the one match script to run.
"""

from dataclasses import dataclass

from chipiron.games.match.match_settings_args import MatchSettingsArgs
from chipiron.games.match.MatchTag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag


@dataclass
class MatchArgs:
    """
    The input arguments needed by the one match script to run

            file_name_player_one (path): The file name for player one. Defaults to 'RecurZipfBase3.yaml'.
            file_name_player_two (path): The file name for player two. Defaults to 'RecurZipfBase3.yaml'.
            file_name_match_setting (path): The file name for the match setting. Defaults to 'setting_cubo.yaml'.
            player_one (dict[Any, Any]): The dictionary for player one. Defaults to an empty dictionary.
            player_two (dict[Any, Any]): The dictionary for player two. Defaults to an empty dictionary.
            match (dict[Any, Any]): The dictionary for the match. Defaults to an empty dictionary.

    """

    player_one: PlayerConfigTag | PlayerArgs = PlayerConfigTag.RECUR_ZIPF_BASE_3
    player_two: PlayerConfigTag | PlayerArgs = PlayerConfigTag.RECUR_ZIPF_BASE_3
    match_setting: MatchConfigTag | MatchSettingsArgs = MatchConfigTag.Cubo
