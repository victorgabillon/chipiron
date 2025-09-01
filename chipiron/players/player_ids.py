"""
This module defines the PlayerConfigTag enumeration, which identifies different player configuration files for the Chipiron chess engine framework. Each tag corresponds to a specific player type, including human interfaces, AI agents, and external engines. The PlayerConfigTag class provides utility methods to determine if a player is human, retrieve the YAML configuration file path for a player, and load player arguments from the configuration file.

Classes:
    PlayerConfigTag (Enum): Enumeration of player configuration tags with methods for player type checking and configuration retrieval.

Methods:
    is_human(): Returns True if the player configuration tag represents a human player.
    get_yaml_file_path(): Returns the file path to the YAML configuration file for the player.
    get_players_args(): Loads and returns player arguments from the corresponding YAML configuration file.

Dependencies:
    - enum
    - importlib.resources
    - parsley_coco
    - chipiron.players.PlayerArgs
    - chipiron.utils.path

"""

from enum import Enum
from importlib.resources import as_file, files

import parsley_coco

from chipiron.players import PlayerArgs
from chipiron.utils import path


class PlayerConfigTag(str, Enum):
    """
    This class is used to identify the player configuration files.
    Each player configuration file should be listed here as a class attribute.
    The class also provides methods to check if a player is human and to get the YAML file path for the player configuration.
    """

    CHIPIRON = "Chipiron"

    GUI_HUMAN = "Gui_Human"
    COMMAND_LINE_HUMAN = "Command_Line_Human"
    SEQUOOL = "Sequool"
    RECUR_ZIPF_BASE_3 = "RecurZipfBase3"
    RECUR_ZIPF_BASE_4 = "RecurZipfBase4"
    UNIFORM_DEPTH = "UniformDepth"
    UNIFORM_DEPTH_2 = "UniformDepth2"
    UNIFORM_DEPTH_3 = "UniformDepth3"
    UNIFORM = "Uniform"
    RANDOM = "Random"
    STOCKFISH = "Stockfish"

    def is_human(self) -> bool:
        """Check if the player is human.
        This method checks if the player is a human player based on the player's configuration tag.
        It returns True if the player is a human player, and False otherwise.

        Returns:
            bool: True if the player is a human player, False otherwise.
        """
        return (
            self is PlayerConfigTag.GUI_HUMAN
            or self is PlayerConfigTag.COMMAND_LINE_HUMAN
        )

    def get_yaml_file_path(self) -> path:
        """Get the YAML file path for the player configuration.
        This method constructs the file path for the player configuration YAML file
        based on the player's configuration tag.
        It returns the file path as a string.

        Returns:
            path: The file path for the player configuration YAML file.
        Raises:
            ValueError: If the player configuration tag is not recognized.
        """
        if self is PlayerConfigTag.CHIPIRON:
            resource = files("chipiron").joinpath(
                "data/players/player_config/chipiron/chipiron.yaml"
            )
        else:
            resource = files("chipiron").joinpath(
                f"data/players/player_config/{self.value}.yaml"
            )

        with as_file(resource) as real_path:
            return str(real_path)  # Or pass it immediately to another function

    def get_players_args(self) -> PlayerArgs:
        """Get the player arguments from the YAML file.
        This method fetches the player arguments from the YAML file
        corresponding to the player's configuration tag.

        Returns:
            PlayerArgs: The player arguments as a dataclass.
        """
        player_args: PlayerArgs = parsley_coco.resolve_yaml_file_to_base_dataclass(
            yaml_path=self.get_yaml_file_path(),
            base_cls=PlayerArgs,
            raise_error_with_nones=False,
            package_name=files("chipiron"),
        )
        return player_args


if __name__ == "__main__":
    a = PlayerConfigTag.COMMAND_LINE_HUMAN
