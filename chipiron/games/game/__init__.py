"""
This module contains the game logic and the game arguments.

The `game` module provides the following classes:
- `ObservableGame`: A class that represents an observable game.
- `Game`: A class that represents a game.
- `GameArgs`: A class that represents the game arguments.
- `GameArgsFactory`: A class that provides a factory for creating game arguments.

The module also defines the `__all__` list, which specifies the names that should be imported when using the `from game import *` syntax.

Example usage:
    from game import ObservableGame, Game, GameArgs, GameArgsFactory
"""

from .game import Game, ObservableGame
from .game_args import GameArgs
from .game_args_factory import GameArgsFactory

__all__ = ["GameArgs", "GameArgsFactory", "Game", "ObservableGame"]
