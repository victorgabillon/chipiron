"""Module for the GameArgsFactory class.

This module defines the GameArgsFactory class, which is responsible for creating game arguments and managing game settings.
"""

import typing

from valanga import SOLO, Color
from valanga.game import Seed

from chipiron import players
from chipiron.core.roles import GameRole
from chipiron.environments.types import GameKind
from chipiron.utils.small_tools import unique_int_from_list

from .game_args import GameArgs

if typing.TYPE_CHECKING:
    from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs


class GameArgsFactory:
    """The GameArgsFactory creates the players and decides the rules.

    So far quite simple
    This class is supposed to be dependent on Match-related classes (contrarily to the GameArgsFactory)
    """

    args_match: "MatchSettingsArgs"
    seed_: int | None
    args_player_one: players.PlayerArgs
    args_player_two: players.PlayerArgs | None
    args_game: GameArgs
    game_number: int

    def __init__(
        self,
        args_match: "MatchSettingsArgs",
        args_player_one: players.PlayerArgs,
        args_player_two: players.PlayerArgs | None,
        seed_: int | None,
        args_game: GameArgs,
    ) -> None:
        """Initialize the instance."""
        self.args_match = args_match
        self.seed_ = seed_
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.game_number = 0

    def generate_game_args(
        self, game_number: int
    ) -> tuple[dict[GameRole, players.PlayerFactoryArgs], GameArgs, Seed | None]:
        """Generate game arguments for a specific game number.

        The returned mapping is role-keyed. Current chess/checkers scheduling is
        still white/black-oriented, while solo games use one real role.

        Args:
            game_number (int): The number of the game.

        Returns:
            tuple[dict[GameRole, players.PlayerFactoryArgs], GameArgs, seed | None]:
                participant assignment for this game, game args, and the merged seed.

        """
        merged_seed: Seed | None = unique_int_from_list([self.seed_, game_number])
        assert merged_seed is not None

        player_one_factory_args = players.PlayerFactoryArgs(
            player_args=self.args_player_one, seed=merged_seed
        )

        if self.args_game.game_kind is GameKind.INTEGER_REDUCTION:
            self.game_number += 1
            return {SOLO: player_one_factory_args}, self.args_game, merged_seed

        assert self.args_player_two is not None
        player_two_factory_args = players.PlayerFactoryArgs(
            player_args=self.args_player_two, seed=merged_seed
        )

        participant_assignment_by_role: dict[GameRole, players.PlayerFactoryArgs]
        if game_number < self.args_match.number_of_games_player_one_white:
            participant_assignment_by_role = {
                Color.WHITE: player_one_factory_args,
                Color.BLACK: player_two_factory_args,
            }
        else:
            participant_assignment_by_role = {
                Color.WHITE: player_two_factory_args,
                Color.BLACK: player_one_factory_args,
            }
        self.game_number += 1

        return participant_assignment_by_role, self.args_game, merged_seed

    def is_match_finished(self) -> bool:
        """Check if the match is finished.

        Returns:
            bool: True if the match is finished, False otherwise.

        """
        return (
            self.game_number
            >= self.args_match.number_of_games_player_one_white
            + self.args_match.number_of_games_player_one_black
        )
