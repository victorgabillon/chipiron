"""
Module to extract messages from players to be shown in the GUI.
"""

from dataclasses import dataclass

import chess

from chipiron.players import PlayerFactoryArgs


@dataclass
class PlayersColorToPlayerMessage:
    """
    Represents a mapping of player colors to GUI information.

    Attributes:
        player_color_to_gui_info (dict[chess.Color, str]): A dictionary mapping player colors to GUI information.
    """

    player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs]
