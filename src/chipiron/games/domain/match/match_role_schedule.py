"""Neutral scheduling helpers for supported match topologies."""

from dataclasses import dataclass

from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs


@dataclass(frozen=True, slots=True)
class TwoRoleMatchSchedule:
    """Schedule quotas for supported 2-role environments.

    The schedule is defined against the ordered role tuple exposed by the
    environment:
    - first role: ``scheduled_roles[0]``
    - second role: ``scheduled_roles[1]``
    """

    number_of_games_player_one_on_first_role: int
    number_of_games_player_one_on_second_role: int

    @property
    def total_games(self) -> int:
        """Return the total number of scheduled games."""
        return (
            self.number_of_games_player_one_on_first_role
            + self.number_of_games_player_one_on_second_role
        )


def build_two_role_match_schedule_from_legacy_settings(
    args_match: MatchSettingsArgs,
) -> TwoRoleMatchSchedule:
    """Translate legacy white/black settings into neutral role-order scheduling.

    This bridge intentionally lives at the match-factory edge so the scheduler
    core no longer depends on white/black vocabulary.
    """
    return TwoRoleMatchSchedule(
        number_of_games_player_one_on_first_role=args_match.number_of_games_player_one_white,
        number_of_games_player_one_on_second_role=args_match.number_of_games_player_one_black,
    )
