"""
This module defines the `MatchArgs` class, which represents the input arguments needed by the one match script to run.
"""

from dataclasses import dataclass, field
from typing import Any

from chipiron.utils import path


@dataclass
class MatchArgs:
    """
    The input arguments needed by the one match script to run
    """
    def __init__(
        self,
        seed: int = 0,
        experiment_output_folder: path | None = None,
        file_name_player_one: path = 'RecurZipfBase3.yaml',
        file_name_player_two: path = 'RecurZipfBase3.yaml',
        file_name_match_setting: path = 'setting_cubo.yaml',
        player_one: dict[Any, Any] = field(default_factory=dict),
        player_two: dict[Any, Any] = field(default_factory=dict),
        match: dict[Any, Any] = field(default_factory=dict),
    ):
        """
        Initialize the MatchArgs object with the specified arguments.

        Args:
            seed (int): The seed for the match.
            experiment_output_folder (path | None): The output folder for the experiment. Defaults to None.
            file_name_player_one (path): The file name for player one. Defaults to 'RecurZipfBase3.yaml'.
            file_name_player_two (path): The file name for player two. Defaults to 'RecurZipfBase3.yaml'.
            file_name_match_setting (path): The file name for the match setting. Defaults to 'setting_cubo.yaml'.
            player_one (dict[Any, Any]): The dictionary for player one. Defaults to an empty dictionary.
            player_two (dict[Any, Any]): The dictionary for player two. Defaults to an empty dictionary.
            match (dict[Any, Any]): The dictionary for the match. Defaults to an empty dictionary.
        """
        self.seed = seed
        self.experiment_output_folder = experiment_output_folder
        self.file_name_player_one = file_name_player_one
        self.file_name_player_two = file_name_player_two
        self.file_name_match_setting = file_name_match_setting
        self.player_one = player_one
        self.player_two = player_two
        self.match = match
