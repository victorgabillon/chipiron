"""
This module defines the `MatchArgs` class, which represents the input arguments needed by the one match script to run.
"""

from dataclasses import dataclass, field
from typing import Any

from chipiron.players.player_ids import PlayerConfigFile
from chipiron.utils import path


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

    file_name_player_one: PlayerConfigFile = PlayerConfigFile.RecurZipfBase3
    file_name_player_two: PlayerConfigFile = PlayerConfigFile.RecurZipfBase3
    file_name_match_setting: path = 'setting_cubo.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict[Any, Any] = field(default_factory=dict)
    player_two: dict[Any, Any] = field(default_factory=dict)
    match: dict[Any, Any] = field(default_factory=dict)
