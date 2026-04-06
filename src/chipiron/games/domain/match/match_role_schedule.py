"""Neutral schedule models for supported match topologies."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from chipiron.core.roles import GameRole


class MatchScheduleType(StrEnum):
    """Tagged schedule kinds supported by the current match layer."""

    SOLO = "solo_match_schedule"
    TWO_ROLE = "two_role_match_schedule"


@dataclass(frozen=True, slots=True)
class SoloMatchSchedule:
    """Schedule for supported 1-role environments."""

    type: Literal[MatchScheduleType.SOLO] = MatchScheduleType.SOLO
    number_of_games: int = 1

    @property
    def total_games(self) -> int:
        """Return the total number of scheduled games."""
        return self.number_of_games


@dataclass(frozen=True, slots=True)
class TwoRoleMatchSchedule:
    """Schedule quotas for supported 2-role environments.

    The schedule is defined against the ordered role tuple exposed by the
    environment:
    - first role: ``scheduled_roles[0]``
    - second role: ``scheduled_roles[1]``
    """

    type: Literal[MatchScheduleType.TWO_ROLE] = MatchScheduleType.TWO_ROLE
    number_of_games_player_one_on_first_role: int = 0
    number_of_games_player_one_on_second_role: int = 0

    @property
    def total_games(self) -> int:
        """Return the total number of scheduled games."""
        return (
            self.number_of_games_player_one_on_first_role
            + self.number_of_games_player_one_on_second_role
        )


type MatchSchedule = SoloMatchSchedule | TwoRoleMatchSchedule


class SoloTopologyRequiresSoloScheduleError(ValueError):
    """Raised when a 1-role environment receives a non-solo schedule."""

    def __init__(
        self,
        *,
        scheduled_roles: tuple[GameRole, ...],
        schedule: MatchSchedule,
    ) -> None:
        """Initialize the topology/schedule mismatch."""
        super().__init__(
            "A 1-role environment requires SoloMatchSchedule, "
            f"got scheduled_roles={scheduled_roles!r} and schedule={schedule!r}."
        )


class TwoRoleTopologyRequiresTwoRoleScheduleError(ValueError):
    """Raised when a 2-role environment receives a non-2-role schedule."""

    def __init__(
        self,
        *,
        scheduled_roles: tuple[GameRole, ...],
        schedule: MatchSchedule,
    ) -> None:
        """Initialize the topology/schedule mismatch."""
        super().__init__(
            "A 2-role environment requires TwoRoleMatchSchedule, "
            f"got scheduled_roles={scheduled_roles!r} and schedule={schedule!r}."
        )


class UnsupportedScheduleValidationTopologyError(ValueError):
    """Raised when schedule validation is attempted for unsupported role counts."""

    def __init__(self, *, scheduled_roles: tuple[GameRole, ...]) -> None:
        """Initialize the unsupported-topology error."""
        super().__init__(
            "Schedule validation supports only 1-role and 2-role environments, "
            f"got scheduled_roles={scheduled_roles!r}."
        )


def validate_schedule_for_supported_topology(
    *,
    scheduled_roles: Sequence[GameRole],
    schedule: MatchSchedule,
) -> MatchSchedule:
    """Validate that a neutral schedule matches a supported role topology."""
    role_tuple = tuple(scheduled_roles)
    if len(role_tuple) == 1:
        if not isinstance(schedule, SoloMatchSchedule):
            raise SoloTopologyRequiresSoloScheduleError(
                scheduled_roles=role_tuple,
                schedule=schedule,
            )
        return schedule
    if len(role_tuple) == 2:
        if not isinstance(schedule, TwoRoleMatchSchedule):
            raise TwoRoleTopologyRequiresTwoRoleScheduleError(
                scheduled_roles=role_tuple,
                schedule=schedule,
            )
        return schedule
    raise UnsupportedScheduleValidationTopologyError(scheduled_roles=role_tuple)
