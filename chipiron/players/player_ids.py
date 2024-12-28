from enum import Enum


class PlayerConfigFile(str, Enum):
    # this list should correspond to files existing in the data/players/player_config folder
    # todo add a test to check the comment above is true

    GuiHuman = 'Gui_Human'
    CommandLineHuman = 'Command_Line_Human'
    Sequool = 'Sequool'
    RecurZipfBase3 = 'RecurZipfBase3'
    UniformDepth = 'UniformDepth'
    Uniform = 'Uniform'
    Random = 'Random'
    Stockfish = 'Stockfish'

    Test_Sequool = 'players_for_test_purposes/Sequool'
    Test_RecurZipfSequool = 'players_for_test_purposes/RecurZipfSequool'
    Test_RecurZipfBase3 = 'players_for_test_purposes/RecurZipfBase3'
    Test_RecurZipfBase4 = 'players_for_test_purposes/RecurZipfBase4'
    Test_Uniform = 'players_for_test_purposes/Uniform'

    def is_human(self) -> bool:
        return self is PlayerConfigFile.GuiHuman or self is PlayerConfigFile.CommandLineHuman


if __name__ == "__main__":
    a = PlayerConfigFile.CommandLineHuman
    print('rrf', a, a.is_human())
