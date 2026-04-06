"""Module to define the MatchSettingsArgs dataclass."""

from dataclasses import dataclass

from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_tag import GameConfigTag


@dataclass
class MatchSettingsArgs:
    """Dataclass to store match settings arguments.

    The ``number_of_games_player_one_white`` and
    ``number_of_games_player_one_black`` fields are legacy config-facing names.
    The scheduler core now translates them once into neutral first-role /
    second-role quotas based on the ordered environment roles.

    """

    number_of_games_player_one_white: int
    number_of_games_player_one_black: int
    game_args: GameConfigTag | GameArgs
