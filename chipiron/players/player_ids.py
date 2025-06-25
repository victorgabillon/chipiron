import os
from enum import Enum

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
    Stockfish = "Stockfish"

    Test_Sequool = "players_for_test_purposes/Sequool"
    Test_RecurZipfSequool = "players_for_test_purposes/RecurZipfSequool"
    Test_RecurZipfBase3 = "players_for_test_purposes/RecurZipfBase3"
    Test_RecurZipfBase4 = "players_for_test_purposes/RecurZipfBase4"
    Test_Uniform = "players_for_test_purposes/Uniform"

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
        path_player: path
        if self is PlayerConfigTag.CHIPIRON:
            path_player = os.path.join(
                "data/players/player_config/chipiron/chipiron.yaml"
            )
        else:
            path_player = os.path.join(
                "data/players/player_config", self.value + ".yaml"
            )
        return path_player

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
        )
        return player_args


if __name__ == "__main__":
    a = PlayerConfigTag.COMMAND_LINE_HUMAN
