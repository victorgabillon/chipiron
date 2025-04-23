import os
from enum import Enum

from chipiron.games.match import MatchSettingsArgs
from chipiron.utils import path
from chipiron.utils.small_tools import fetch_args_modify_and_convert


class MatchConfigTag(str, Enum):
    # this list should correspond to files existing in the data/settings/OneMatch folder

    Cubo = "setting_cubo"
    Duda = "setting_duda"
    Jime = "setting_jime"
    Tron = "setting_tron"

    def get_yaml_file_path(self) -> path:
        path_player: path = os.path.join("data/settings/OneMatch", self.value + ".yaml")
        return path_player

    def get_match_settings_args(self) -> MatchSettingsArgs:
        match_args: MatchSettingsArgs = fetch_args_modify_and_convert(
            path_to_file=self.get_yaml_file_path(), dataclass_name=MatchSettingsArgs
        )
        return match_args
