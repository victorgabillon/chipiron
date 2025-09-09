"""Enum representing available game configuration tags."""

from enum import Enum
from importlib.resources import as_file, files

import parsley_coco

from chipiron.games.game.game_args import GameArgs
from chipiron.utils import path


class GameConfigTag(str, Enum):
    """
    Enum representing available game configuration tags.
    """

    NAVO = "setting_navo"
    PAIN = "setting_pain"

    def get_yaml_file_path(self) -> path:
        """Returns the file path to the YAML configuration file associated with the tag.

        Returns:
            path: The file path to the YAML configuration file.
        """
        resource = files("chipiron").joinpath(
            "data/settings/GameSettings", self.value + ".yaml"
        )
        with as_file(resource) as path_player:
            return path_player

    def get_match_settings_args(self) -> GameArgs:
        """Parses the YAML configuration file and returns its contents as a GameArgs instance.

        Returns:
            GameArgs: The parsed match settings.
        """
        game_args: GameArgs = parsley_coco.resolve_yaml_file_to_base_dataclass(
            yaml_path=str(self.get_yaml_file_path()), base_cls=GameArgs
        )
        return game_args
