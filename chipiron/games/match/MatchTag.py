import os
from enum import Enum

import parsley_coco

from chipiron.games.match.match_settings_args import MatchSettingsArgs
from chipiron.utils import path


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
        match_args: MatchSettingsArgs = (
            parsley_coco.resolve_yaml_file_to_base_dataclass(
                yaml_path=self.get_yaml_file_path(), base_cls=MatchSettingsArgs
            )
        )
        return match_args
