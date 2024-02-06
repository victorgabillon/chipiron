from dataclasses import dataclass
import os


@dataclass
class MatchSettingsArgs:
    number_of_games_player_one_white: int
    number_of_games_player_one_black: int
    game_setting_file: str | bytes | os.PathLike
