from dataclasses import dataclass

import chess

from chipiron.players import PlayerFactoryArgs
from chipiron.players.move_selector.treevalue import TreeAndValuePlayerArgs
from chipiron.players.move_selector.treevalue.stopping_criterion import TreeMoveLimitArgs


@dataclass
class PlayersColorToPlayerMessage:
    player_color_to_gui_info: dict[chess.Color, str]


def extract_message_from_players(
        player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs],
) -> PlayersColorToPlayerMessage:
    player_color_to_gui_info: dict[chess.Color, str] = {
        color: extract_message_from_player(player_factory_args) for color, player_factory_args in
        player_color_to_factory_args.items()
    }
    return PlayersColorToPlayerMessage(player_color_to_gui_info=player_color_to_gui_info)


def extract_message_from_player(
        player: PlayerFactoryArgs
) -> str:
    name: str = player.player_args.name
    tree_move_limit: str | int = ''
    if isinstance(player.player_args.main_move_selector, TreeAndValuePlayerArgs):
        if isinstance(player.player_args.main_move_selector.stopping_criterion, TreeMoveLimitArgs):
            tree_move_limit = player.player_args.main_move_selector.stopping_criterion.tree_move_limit

    return f'{name} ({tree_move_limit})'
