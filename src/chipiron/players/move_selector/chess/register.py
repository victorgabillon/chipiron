"""Register chess-specific move selector factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.types import GameKind
from chipiron.players.move_selector.human import (
    CommandLineHumanMoveSelector,
    CommandLineHumanPlayerArgs,
)
from chipiron.players.move_selector.registry import (
    register_game_specific_selector_factory,
)
from chipiron.players.move_selector.stockfish_args import StockfishSelectorArgs
from chipiron.players.move_selector.stockfish_selector import create_stockfish_selector

if TYPE_CHECKING:
    import random

    from valanga import Dynamics
    from valanga.policy import BranchSelector

    from chipiron.players.move_selector.move_selector_args import (
        NonTreeMoveSelectorArgs,
    )


ChessSelectorArgs = CommandLineHumanPlayerArgs | StockfishSelectorArgs


class UnsupportedChessSelectorArgsError(ValueError):
    """Raised when chess selector args are not supported by the chess registry."""

    def __init__(self, args: object) -> None:
        """Initialize the error with the unsupported args object."""
        super().__init__(f"Unsupported chess selector args: {type(args)}")


def _chess_specific_factory(
    args: NonTreeMoveSelectorArgs,
    dynamics: Dynamics[Any],
    random_generator: random.Random,
) -> BranchSelector[Any]:
    """Create a chess-specific move selector from args."""
    _ = random_generator

    match args:
        case CommandLineHumanPlayerArgs():
            return CommandLineHumanMoveSelector(dynamics=dynamics)
        case StockfishSelectorArgs():
            return create_stockfish_selector(
                depth=args.depth,
                time_limit=args.time_limit,
            )
        case _:
            raise UnsupportedChessSelectorArgsError(args)


def register_chess_move_selectors() -> None:
    """Register chess-specific move selector factories."""
    register_game_specific_selector_factory(GameKind.CHESS, _chess_specific_factory)
