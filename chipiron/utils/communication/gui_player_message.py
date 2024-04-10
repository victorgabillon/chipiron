"""
Module to extract messages from players to be shown in the GUI.
"""
from dataclasses import dataclass

import chess

from chipiron.players import PlayerFactoryArgs
from chipiron.players.move_selector.treevalue import TreeAndValuePlayerArgs
from chipiron.players.move_selector.treevalue.stopping_criterion import TreeMoveLimitArgs


@dataclass
class PlayersColorToPlayerMessage:
    """
    Represents a mapping of player colors to GUI information.

    Attributes:
        player_color_to_gui_info (dict[chess.Color, str]): A dictionary mapping player colors to GUI information.
    """
    player_color_to_gui_info: dict[chess.Color, str]


def extract_message_from_players(
        player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs],
) -> PlayersColorToPlayerMessage:
    """
    Extracts messages from players to be shown in the GUI.

    Args:
        player_color_to_factory_args (dict[chess.Color, PlayerFactoryArgs]): A dictionary mapping player colors to
            their factory arguments.

    Returns:
        PlayersColorToPlayerMessage: An object containing the extracted messages for each player color.
    """
    player_color_to_gui_info: dict[chess.Color, str] = {
        color: extract_message_from_player(player_factory_args) for color, player_factory_args in
        player_color_to_factory_args.items()
    }
    return PlayersColorToPlayerMessage(player_color_to_gui_info=player_color_to_gui_info)


def extract_message_from_player(
        player: PlayerFactoryArgs
) -> str:
    """
    Extracts a message from a player to be shown in the GUI.

    Args:
        player (PlayerFactoryArgs): The factory arguments for the player.

    Returns:
        str: The extracted message.
    """
    name: str = player.player_args.name
    tree_move_limit: str | int = ''
    if isinstance(player.player_args.main_move_selector, TreeAndValuePlayerArgs):
        if isinstance(player.player_args.main_move_selector.stopping_criterion, TreeMoveLimitArgs):
            tree_move_limit = player.player_args.main_move_selector.stopping_criterion.tree_move_limit

    return f'{name} ({tree_move_limit})'
