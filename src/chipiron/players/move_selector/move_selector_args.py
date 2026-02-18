"""Move selector args protocol and concrete args unions for YAML/config parsing."""

from typing import Any, Protocol

from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs

from . import human
from .move_selector_types import MoveSelectorTypes
from .random_args import RandomSelectorArgs
from .stockfish_args import StockfishSelectorArgs


class MoveSelectorArgs(Protocol):
    """Protocol for arguments for MoveSelector construction."""

    type: MoveSelectorTypes

    def is_human(self) -> bool:
        """Return whether human."""
        return self.type.is_human()


NonTreeMoveSelectorArgs = (
    human.CommandLineHumanPlayerArgs | RandomSelectorArgs | StockfishSelectorArgs
)

AnyMoveSelectorArgs = (
    TreeAndValueAppArgs | NonTreeMoveSelectorArgs | human.GuiHumanPlayerArgs
)


_MOVE_SELECTOR_ARGS_TYPES = (
    TreeAndValueAppArgs,
    human.CommandLineHumanPlayerArgs,
    human.GuiHumanPlayerArgs,
    RandomSelectorArgs,
    StockfishSelectorArgs,
)

_MOVE_SELECTOR_ARGS_BY_TYPE: dict[MoveSelectorTypes, type[Any]] = {
    MoveSelectorTypes.TREE_AND_VALUE: TreeAndValueAppArgs,
    MoveSelectorTypes.COMMAND_LINE_HUMAN: human.CommandLineHumanPlayerArgs,
    MoveSelectorTypes.GUI_HUMAN: human.GuiHumanPlayerArgs,
    MoveSelectorTypes.RANDOM: RandomSelectorArgs,
    MoveSelectorTypes.STOCKFISH: StockfishSelectorArgs,
}
