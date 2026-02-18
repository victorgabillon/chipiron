"""Provide registration and lookup for game-specific move selector factories."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from valanga import Dynamics, TurnState
from valanga.policy import BranchSelector

from chipiron.players.move_selector.move_selector_args import NonTreeMoveSelectorArgs

if TYPE_CHECKING:
    from chipiron.environments.types import GameKind

StateT = TurnState

# Handler for "game-specific" selectors (like Stockfish for chess).
type GameSpecificSelectorFactory = Callable[
    [
        NonTreeMoveSelectorArgs,
        Dynamics[Any],
        random.Random,
    ],  # (args, dynamics, random_generator)
    BranchSelector[Any],
]

_FACTORIES: dict[GameKind, GameSpecificSelectorFactory] = {}


def register_game_specific_selector_factory(
    game_kind: GameKind,
    factory: GameSpecificSelectorFactory,
) -> None:
    """Register a factory function for game-specific move selectors."""
    _FACTORIES[game_kind] = factory


def get_game_specific_selector_factory(
    game_kind: GameKind,
) -> GameSpecificSelectorFactory | None:
    """Get the registered factory for a given game kind, or None if not registered."""
    return _FACTORIES.get(game_kind)
