import os
from enum import Enum

from chipiron.players import PlayerArgs
from chipiron.utils import path
from chipiron.utils.small_tools import fetch_args_modify_and_convert


class PlayerConfigTag(str, Enum):
    # this list should correspond to files existing in the data/players/player_config folder
    # todo add a test to check the comment above is true

    Chipiron = "Chipiron"

    GuiHuman = "Gui_Human"
    CommandLineHuman = "Command_Line_Human"
    Sequool = "Sequool"
    RecurZipfBase3 = "RecurZipfBase3"
    RecurZipfBase4 = "RecurZipfBase4"
    UniformDepth = "UniformDepth"
    UniformDepth2 = "UniformDepth2"
    UniformDepth3 = "UniformDepth3"
    Uniform = "Uniform"
    Random = "Random"
    Stockfish = "Stockfish"

    Test_Sequool = "players_for_test_purposes/Sequool"
    Test_RecurZipfSequool = "players_for_test_purposes/RecurZipfSequool"
    Test_RecurZipfBase3 = "players_for_test_purposes/RecurZipfBase3"
    Test_RecurZipfBase4 = "players_for_test_purposes/RecurZipfBase4"
    Test_Uniform = "players_for_test_purposes/Uniform"

    def is_human(self) -> bool:
        return (
            self is PlayerConfigTag.GuiHuman or self is PlayerConfigTag.CommandLineHuman
        )

    def get_yaml_file_path(self) -> path:
        path_player: path
        if self is PlayerConfigTag.Chipiron:
            path_player = os.path.join(
                "data/players/player_config/chipiron/chipiron.yaml"
            )
        else:
            path_player = os.path.join(
                "data/players/player_config", self.value + ".yaml"
            )
        return path_player

    def get_players_args(self) -> PlayerArgs:
        player_args: PlayerArgs = fetch_args_modify_and_convert(
            path_to_file=self.get_yaml_file_path(), dataclass_name=PlayerArgs
        )
        return player_args


if __name__ == "__main__":
    a = PlayerConfigTag.CommandLineHuman
