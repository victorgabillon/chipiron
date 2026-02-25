"""Document the module defines the MatchConfigTag enumeration for referencing specific match configuration YAML files.

in the 'data/settings/OneMatch' directory. Each tag corresponds to a configuration file and provides methods
to retrieve the file path and parse its contents into a MatchSettingsArgs dataclass.

Classes:
    MatchConfigTag (Enum): Enum representing available match configuration tags.

Methods:
    get_yaml_file_path() -> path:
        Returns the file path to the YAML configuration file associated with the tag.

    get_match_settings_args() -> MatchSettingsArgs:
        Parses the YAML configuration file and returns its contents as a MatchSettingsArgs instance.

"""

from enum import StrEnum
from importlib.resources import as_file, files

import parsley

from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs
from chipiron.utils import MyPath


class MatchConfigTag(StrEnum):
    """Enum representing available match configuration tags."""

    # this list should correspond to files existing in the data/settings/OneMatch folder

    CUBO = "setting_cubo"
    DUDA = "setting_duda"
    JIME = "setting_jime"
    TRON = "setting_tron"

    def get_yaml_file_path(self) -> MyPath:
        """Return the file path to the YAML configuration file associated with the tag.

        Returns:
            path: The file path to the YAML configuration file.

        """
        resource = files("chipiron").joinpath(
            "data/settings/OneMatch", self.value + ".yaml"
        )
        with as_file(resource) as path_player:
            return path_player

    def get_match_settings_args(self) -> MatchSettingsArgs:
        """Parse the YAML configuration file and returns its contents as a MatchSettingsArgs instance.

        Returns:
            MatchSettingsArgs: The parsed match settings.

        """
        match_args: MatchSettingsArgs = parsley.resolve_yaml_file_to_base_dataclass(
            yaml_path=str(self.get_yaml_file_path()),
            base_cls=MatchSettingsArgs,
            raise_error_with_nones=False,
        )
        return match_args
