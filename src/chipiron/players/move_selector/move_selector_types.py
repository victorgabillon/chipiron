"""Document the module defines an enumeration class for Move Selector Types.

The MoveSelectorTypes class is a subclass of the str class and the Enum class.
It represents different types of move selectors that can be used in a game.

Attributes:
    Random: Represents a random move selector.
    TreeAndValue: Represents a move selector based on tree and value.
    Stockfish: Represents a move selector using the Stockfish engine.
    CommandLineHuman: Represents a move selector for a human player using the command line.
    GuiHuman: Represents a move selector for a human player using a graphical user interface.

"""

from enum import StrEnum

from anemone.factory import TREE_AND_VALUE_LITERAL_STRING


class MoveSelectorTypes(StrEnum):
    """Enumeration class representing different types of move selectors."""

    RANDOM = "Random"
    TREE_AND_VALUE = TREE_AND_VALUE_LITERAL_STRING
    STOCKFISH = "Stockfish"
    COMMAND_LINE_HUMAN = "CommandLineHuman"
    GUI_HUMAN = "GuiHuman"

    def is_human(self) -> bool:
        """Determine if the move selector type represents a human player.

        Returns:
            bool: True if the move selector type is either GuiHuman or CommandLineHuman, False otherwise.

        """
        return (
            self is MoveSelectorTypes.GUI_HUMAN
            or self is MoveSelectorTypes.COMMAND_LINE_HUMAN
        )
