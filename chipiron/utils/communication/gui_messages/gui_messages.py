"""Backwards-compatible import surface for GUI protocol types.

Canonical message types live in `chipiron.displays.gui_protocol`.
This module re-exports them to avoid touching all call sites at once,
while keeping a couple helper functions used by factories.
"""

from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from valanga import Color

from chipiron.displays.gui_protocol import PlayerUiInfo, UpdPlayersInfo
from chipiron.players import PlayerFactoryArgs

def format_player_label(player: PlayerFactoryArgs) -> str:
    name: str = player.player_args.name

    tree_branch_limit: str | int = ""
    sel = player.player_args.main_move_selector

    if isinstance(sel, TreeAndValuePlayerArgs):
        stop = sel.stopping_criterion
        if isinstance(stop, TreeBranchLimitArgs):
            tree_branch_limit = stop.tree_branch_limit

    return f"{name} ({tree_branch_limit})"


def make_players_info_payload(
    player_color_to_factory_args: dict[Color, PlayerFactoryArgs],
) -> UpdPlayersInfo:
    w = player_color_to_factory_args[Color.WHITE]
    b = player_color_to_factory_args[Color.BLACK]

    return UpdPlayersInfo(
        white=PlayerUiInfo(
            label=format_player_label(w), is_human=w.player_args.is_human()
        ),
        black=PlayerUiInfo(
            label=format_player_label(b), is_human=b.player_args.is_human()
        ),
    )
