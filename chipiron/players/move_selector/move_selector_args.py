"""
This module defines the MoveSelectorArgs protocol and helpers for move selector arguments.
"""

from typing import Any, Protocol, TypeAlias

from chipiron.players.move_selector.factory import NonTreeMoveSelectorArgs
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs

from . import human, random, stockfish
from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """Protocol for arguments for MoveSelector construction."""

    type: MoveSelectorTypes

    def is_human(self) -> bool:
        return self.type.is_human()


AnyMoveSelectorArgs: TypeAlias = (
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
    MoveSelectorTypes.TreeAndValue: TreeAndValueAppArgs,
    MoveSelectorTypes.CommandLineHuman: human.CommandLineHumanPlayerArgs,
    MoveSelectorTypes.GuiHuman: human.GuiHumanPlayerArgs,
    MoveSelectorTypes.Random: random.Random,
    MoveSelectorTypes.Stockfish: stockfish.StockfishPlayer,
}
