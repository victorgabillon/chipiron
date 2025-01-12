"""
This module defines an enumeration class for Move Selector Types.

The MoveSelectorTypes class is a subclass of the str class and the Enum class.
It represents different types of move selectors that can be used in a game.

Attributes:
    Random: Represents a random move selector.
    TreeAndValue: Represents a move selector based on tree and value.
    Stockfish: Represents a move selector using the Stockfish engine.
    CommandLineHuman: Represents a move selector for a human player using the command line.
    GuiHuman: Represents a move selector for a human player using a graphical user interface.
"""

from enum import Enum


class MoveSelectorTypes(str, Enum):
    """
    Enumeration class representing different types of move selectors.
    """

    Random = "Random"
    TreeAndValue = "TreeAndValue"
    Stockfish = "Stockfish"
    CommandLineHuman = "CommandLineHuman"
    GuiHuman = "GuiHuman"

    def is_human(self) -> bool:
        return (
            self is MoveSelectorTypes.GuiHuman
            or self is MoveSelectorTypes.CommandLineHuman
        )
