"""Game-kind specific defaults for GUI launcher state."""

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag

from .models import ArgsChosenByUser


def apply_game_kind_defaults(args: ArgsChosenByUser) -> None:
    """Apply game-kind defaults when switching game kind in the launcher."""
    if args.game_kind is GameKind.CHECKERS:
        args.player_type_white = PlayerConfigTag.GUI_HUMAN
        args.strength_white = None
        args.player_type_black = PlayerConfigTag.GUI_HUMAN
        args.strength_black = None
    elif args.game_kind is GameKind.CHESS:
        args.player_type_white = PlayerConfigTag.RECUR_ZIPF_BASE_3
        args.strength_white = 1
        args.player_type_black = PlayerConfigTag.RECUR_ZIPF_BASE_3
        args.strength_black = 1
