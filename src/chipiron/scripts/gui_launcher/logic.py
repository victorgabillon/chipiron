"""Game-kind specific defaults for launcher state."""

from .models import ArgsChosenByUser
from .registries import launcher_spec_for_game


def apply_game_kind_defaults(args: ArgsChosenByUser) -> None:
    """Reset launcher selections from the selected game's declarative spec."""
    launcher_spec = launcher_spec_for_game(args.game_kind)
    args.participants = list(launcher_spec.default_participants)
    args.starting_position_key = launcher_spec.default_starting_position_key
