"""Create per-game participant assignments from validated role scheduling."""

from valanga.game import Seed

from chipiron import players
from chipiron.core.roles import GameRole
from chipiron.games.domain.match.match_role_schedule import ValidatedMatchPlan
from chipiron.utils.small_tools import unique_int_from_list

from .game_args import GameArgs


class GameArgsFactory:
    """Create role-keyed participant assignments from a validated match plan."""

    seed_: int | None
    args_player_one: players.PlayerArgs
    args_player_two: players.PlayerArgs | None
    args_game: GameArgs
    match_plan: ValidatedMatchPlan
    game_number: int

    def __init__(
        self,
        args_player_one: players.PlayerArgs,
        args_player_two: players.PlayerArgs | None,
        seed_: int | None,
        args_game: GameArgs,
        match_plan: ValidatedMatchPlan,
    ) -> None:
        """Initialize the instance."""
        self.seed_ = seed_
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.match_plan = match_plan
        self.game_number = 0

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return the validated ordered participant identifiers for this match."""
        return self.match_plan.participant_ids

    def generate_game_args(
        self, game_number: int
    ) -> tuple[dict[GameRole, players.PlayerFactoryArgs], GameArgs, Seed | None]:
        """Generate game arguments for a specific game number.

        The returned mapping is role-keyed. Scheduling is fully driven by the
        validated match plan rather than by ad hoc topology checks.

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

        participant_factory_args_by_index: tuple[players.PlayerFactoryArgs, ...]
        if self.match_plan.requires_second_participant:
            assert self.args_player_two is not None
            player_two_factory_args = players.PlayerFactoryArgs(
                player_args=self.args_player_two, seed=merged_seed
            )
            participant_factory_args_by_index = (
                player_one_factory_args,
                player_two_factory_args,
            )
        else:
            participant_factory_args_by_index = (player_one_factory_args,)

        participant_assignment_by_role = {
            role: participant_factory_args_by_index[participant_index]
            for role, participant_index in zip(
                self.match_plan.scheduled_roles,
                self.match_plan.participant_indexes_for_roles(game_number),
                strict=True,
            )
        }
        self.game_number += 1

        return participant_assignment_by_role, self.args_game, merged_seed

    def is_match_finished(self) -> bool:
        """Check if the match is finished.

        Returns:
            bool: True if the match is finished, False otherwise.

        """
        return self.game_number >= self.match_plan.total_games
