"""
Module for player arguments.
"""

from dataclasses import dataclass
from typing import Protocol

from chipiron.players.move_selector.move_selector_args import (
    AnyMoveSelectorArgs,
)

from .move_selector.move_selector_types import MoveSelectorTypes


class HasMoveSelectorType(Protocol):
    """Protocol for objects exposing a move-selector type."""

    @property
    def type(self) -> MoveSelectorTypes:
        """Return the move-selector type."""
        ...


@dataclass
class PlayerArgs:
    """Represents the arguments for a player.

    Attributes:
        name (str): The name of the player.
        main_move_selector (AnyMoveSelectorArgs): The main move selector for the player.
        syzygy_play (bool): Whether to play with syzygy when possible.
    """

    name: str
    main_move_selector: AnyMoveSelectorArgs
    syzygy_play: bool

    def is_human(self) -> bool:
        """Check if the player is a human player.

        Returns:
             : bool: True if the player is a human player, False otherwise.
        """
        return MoveSelectorTypes(self.main_move_selector.type).is_human()


@dataclass
class PlayerFactoryArgs:
    """A class representing the arguments for creating a player factory.

    Attributes:
        player_args (PlayerArgs): The arguments for the player.
        seed (int): The seed value for random number generation.
    """

    player_args: PlayerArgs
    seed: int


AnyPlayerArgs = PlayerArgs
