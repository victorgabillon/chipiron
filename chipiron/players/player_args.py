"""
Module for player arguments.
"""
from dataclasses import dataclass

from . import move_selector


@dataclass
class PlayerArgs:
    """Represents the arguments for a player.

    Attributes:
        name (str): The name of the player.
        main_move_selector (move_selector.AllMoveSelectorArgs): The main move selector for the player.
        syzygy_play (bool): Whether to play with syzygy when possible.
    """
    name: str
    main_move_selector: move_selector.AllMoveSelectorArgs
    syzygy_play: bool


@dataclass
class PlayerFactoryArgs:
    """A class representing the arguments for creating a player factory.

    Attributes:
        player_args (PlayerArgs): The arguments for the player.
        seed (int): The seed value for random number generation.
    """
    player_args: PlayerArgs
    seed: int
