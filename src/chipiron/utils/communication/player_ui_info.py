"""Helpers for GUI player labels and player-info payloads."""

from collections.abc import Sequence

from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from valanga import Color

from chipiron.core.roles import GameRole, RoleAssignment, format_game_role
from chipiron.displays.gui_protocol import ParticipantUiInfo, UpdParticipantsInfo
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


def make_participants_info_payload(
    participant_factory_args_by_role: RoleAssignment[PlayerFactoryArgs],
    *,
    role_order: Sequence[GameRole] | None = None,
) -> UpdParticipantsInfo:
    """Create a generic participant-info payload ordered by game roles."""
    ordered_roles = (
        role_order
        if role_order is not None
        else tuple(participant_factory_args_by_role.keys())
    )
    return UpdParticipantsInfo(
        participants=tuple(
            ParticipantUiInfo(
                role=role,
                role_label=format_game_role(role),
                label=format_player_label(participant_factory_args_by_role[role]),
                is_human=participant_factory_args_by_role[role].player_args.is_human(),
            )
            for role in ordered_roles
        )
    )


def make_players_info_payload(
    participant_factory_args_by_color: dict[Color, PlayerFactoryArgs],
) -> UpdParticipantsInfo:
    """Backward-compatible wrapper for the current white/black scheduling flow."""
    role_assignments: dict[GameRole, PlayerFactoryArgs] = {
        Color.WHITE: participant_factory_args_by_color[Color.WHITE],
        Color.BLACK: participant_factory_args_by_color[Color.BLACK],
    }
    return make_participants_info_payload(
        participant_factory_args_by_role=role_assignments,
        role_order=(Color.WHITE, Color.BLACK),
    )
