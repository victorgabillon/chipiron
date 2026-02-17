"""Document the module defines the MoveSelectorArgs protocol and helpers for move selector arguments."""

from typing import Any, Protocol

from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs

from . import human, random, stockfish
from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """Protocol for arguments for MoveSelector construction."""

    type: MoveSelectorTypes

    def is_human(self) -> bool:
        """Return whether human."""
        return self.type.is_human()


NonTreeMoveSelectorArgs = (
    human.CommandLineHumanPlayerArgs | random.Random[Any] | stockfish.StockfishPlayer
)

AnyMoveSelectorArgs = (
    TreeAndValueAppArgs | NonTreeMoveSelectorArgs | human.GuiHumanPlayerArgs
)


_MOVE_SELECTOR_ARGS_TYPES = (
    TreeAndValueAppArgs,
    human.CommandLineHumanPlayerArgs,
    human.GuiHumanPlayerArgs,
    random.Random,
    stockfish.StockfishPlayer,
)

_MOVE_SELECTOR_ARGS_BY_TYPE: dict[MoveSelectorTypes, type[Any]] = {
    MoveSelectorTypes.TREE_AND_VALUE: TreeAndValueAppArgs,
    MoveSelectorTypes.COMMAND_LINE_HUMAN: human.CommandLineHumanPlayerArgs,
    MoveSelectorTypes.GUI_HUMAN: human.GuiHumanPlayerArgs,
    MoveSelectorTypes.RANDOM: random.Random,
    MoveSelectorTypes.STOCKFISH: stockfish.StockfishPlayer,
}
