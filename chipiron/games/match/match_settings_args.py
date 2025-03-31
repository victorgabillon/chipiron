"""Module to define the MatchSettingsArgs dataclass."""

from dataclasses import dataclass

from chipiron.utils import path


@dataclass
class MatchSettingsArgs:
    """Dataclass to store match settings arguments.

    Args:
        number_of_games_player_one_white (int): The number of games player one plays as white.
        number_of_games_player_one_black (int): The number of games player one plays as black.
        game_setting_file (path): The file path to the game setting file.
    """

    number_of_games_player_one_white: int
    number_of_games_player_one_black: int
    game_setting_file: path
