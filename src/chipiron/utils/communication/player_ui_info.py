"""Helpers for GUI player labels and player-info payloads."""

from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from valanga import Color

from chipiron.displays.gui_protocol import PlayerUiInfo, UpdPlayersInfo
from chipiron.players import PlayerFactoryArgs
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs


def format_player_label(player: PlayerFactoryArgs) -> str:
    """Format player label."""
    name: str = player.player_args.name

    tree_branch_limit: str | int = ""
    sel = player.player_args.main_move_selector

    if isinstance(sel, TreeAndValueAppArgs):
        stop = sel.anemone_args.stopping_criterion
        if isinstance(stop, TreeBranchLimitArgs):
            tree_branch_limit = stop.tree_branch_limit

    return f"{name} ({tree_branch_limit})"


def make_players_info_payload(
    player_color_to_factory_args: dict[Color, PlayerFactoryArgs],
) -> UpdPlayersInfo:
    """Create players info payload."""
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
