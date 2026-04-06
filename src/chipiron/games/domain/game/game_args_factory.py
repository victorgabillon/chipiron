"""Create per-game participant assignments from validated role scheduling."""

from valanga.game import Seed

from chipiron import players
from chipiron.core.roles import GameRole
from chipiron.games.domain.match.match_role_schedule import (
    MatchSchedule,
    SoloMatchSchedule,
    TwoRoleMatchSchedule,
    validate_schedule_for_supported_topology,
)
from chipiron.utils.small_tools import unique_int_from_list

from .game_args import GameArgs


class UnsupportedSchedulingTopologyError(ValueError):
    """Raised when GameArgsFactory receives an unsupported role topology."""

    def __init__(self, *, scheduled_roles: tuple[GameRole, ...]) -> None:
        """Initialize the error with the unsupported scheduling roles."""
        super().__init__(
            "GameArgsFactory supports only 1-role and 2-role scheduling, "
            f"got scheduled_roles={scheduled_roles!r}."
        )


class GameArgsFactory:
    """Create role-keyed participant assignments for each scheduled game."""

    seed_: int | None
    args_player_one: players.PlayerArgs
    args_player_two: players.PlayerArgs | None
    args_game: GameArgs
    scheduled_roles: tuple[GameRole, ...]
    schedule: MatchSchedule
    game_number: int

    def __init__(
        self,
        args_player_one: players.PlayerArgs,
        args_player_two: players.PlayerArgs | None,
        seed_: int | None,
        args_game: GameArgs,
        scheduled_roles: tuple[GameRole, ...],
        schedule: MatchSchedule,
    ) -> None:
        """Initialize the instance."""
        self.seed_ = seed_
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.scheduled_roles = scheduled_roles
        self.schedule = schedule
        self.game_number = 0

    def generate_game_args(
        self, game_number: int
    ) -> tuple[dict[GameRole, players.PlayerFactoryArgs], GameArgs, Seed | None]:
        """Generate game arguments for a specific game number.

        The returned mapping is role-keyed. Scheduling is driven by the validated
        environment role topology: one real role for solo games, or a neutral
        first-role/second-role schedule for 2-role games.

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

        if len(self.scheduled_roles) == 1:
            validated_schedule = validate_schedule_for_supported_topology(
                scheduled_roles=self.scheduled_roles,
                schedule=self.schedule,
            )
            assert isinstance(validated_schedule, SoloMatchSchedule)
            self.game_number += 1
            return (
                {self.scheduled_roles[0]: player_one_factory_args},
                self.args_game,
                merged_seed,
            )

        if len(self.scheduled_roles) != 2:
            raise UnsupportedSchedulingTopologyError(
                scheduled_roles=self.scheduled_roles
            )
        validated_schedule = validate_schedule_for_supported_topology(
            scheduled_roles=self.scheduled_roles,
            schedule=self.schedule,
        )
        assert isinstance(validated_schedule, TwoRoleMatchSchedule)

        assert self.args_player_two is not None
        player_two_factory_args = players.PlayerFactoryArgs(
            player_args=self.args_player_two, seed=merged_seed
        )
        first_role, second_role = self.scheduled_roles

        participant_assignment_by_role: dict[GameRole, players.PlayerFactoryArgs]
        if (
            game_number
            < validated_schedule.number_of_games_player_one_on_first_role
        ):
            participant_assignment_by_role = {
                first_role: player_one_factory_args,
                second_role: player_two_factory_args,
            }
        else:
            participant_assignment_by_role = {
                first_role: player_two_factory_args,
                second_role: player_one_factory_args,
            }
        self.game_number += 1

        return participant_assignment_by_role, self.args_game, merged_seed

    def is_match_finished(self) -> bool:
        """Check if the match is finished.

        Returns:
            bool: True if the match is finished, False otherwise.

        """
        return self.game_number >= self.schedule.total_games
