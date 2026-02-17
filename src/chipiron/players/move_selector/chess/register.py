"""Register chess-specific move selector factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.types import GameKind
from chipiron.players.move_selector.registry import (
    register_game_specific_selector_factory,
)
from chipiron.players.move_selector.stockfish import StockfishPlayer

if TYPE_CHECKING:
    import random

    from valanga import Dynamics
    from valanga.policy import BranchSelector


class UnsupportedChessSelectorArgsError(ValueError):
    """Raised when chess selector args are not supported by the chess registry."""

    def __init__(self, args: Any) -> None:
        """Initialize the error with the unsupported args object."""
        super().__init__(f"Unsupported chess selector args: {type(args)}")


def _chess_specific_factory(
    args: Any,
    dynamics: Dynamics[Any],
    random_generator: random.Random,
) -> BranchSelector[Any]:
    """Create a chess-specific move selector from args."""
    _ = dynamics
    _ = random_generator

    match args:
        case StockfishPlayer():
            # StockfishPlayer already implements BranchSelector[ChessState]
            return args
        case _:
            raise UnsupportedChessSelectorArgsError(args)


def register_chess_move_selectors() -> None:
    """Register chess-specific move selector factories."""
    register_game_specific_selector_factory(GameKind.CHESS, _chess_specific_factory)
