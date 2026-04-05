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
    participant_factory_args_by_color: dict[Color, PlayerFactoryArgs],
) -> UpdPlayersInfo:
    """Create the current white/black player info payload.

    The input mapping is still color-keyed because the GUI payload remains
    explicitly white/black in the current runtime.
    """
    w = participant_factory_args_by_color[Color.WHITE]
    b = participant_factory_args_by_color[Color.BLACK]

    return UpdPlayersInfo(
        white=PlayerUiInfo(
            label=format_player_label(w), is_human=w.player_args.is_human()
        ),
        black=PlayerUiInfo(
            label=format_player_label(b), is_human=b.player_args.is_human()
        ),
    )
