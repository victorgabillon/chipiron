from dataclasses import dataclass

import chess

from chipiron.players import Player
from chipiron.players.move_selector.treevalue.stopping_criterion import TreeMoveLimitArgs
from chipiron.players.move_selector.treevalue.tree_and_value_player import TreeAndValueMoveSelector


@dataclass
class PlayersColorToPlayerMessage:
    player_color_to_gui_info: dict[chess.Color, str]


def extract_message_from_players(
        player_color_to_player: dict[chess.Color, Player]
) -> PlayersColorToPlayerMessage:
    player_color_to_gui_info: dict[chess.Color, str] = {color: extract_message_from_player(player) for color, player in
                                                        player_color_to_player.items()}
    return PlayersColorToPlayerMessage(player_color_to_gui_info=player_color_to_gui_info)


def extract_message_from_player(
        player: Player
) -> str:
    name: str = player.id
    tree_move_limit: str | int = ''
    if isinstance(player.main_move_selector, TreeAndValueMoveSelector):
        if isinstance(player.main_move_selector.stopping_criterion_args, TreeMoveLimitArgs):
            tree_move_limit = player.main_move_selector.stopping_criterion_args.tree_move_limit

    return f'{name} ({tree_move_limit})'
