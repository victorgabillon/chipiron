"""Neutral schedules and validated match plans for supported topologies."""

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


class ParticipantRoleTopologyMismatchError(ValueError):
    """Raised when configured participants do not match environment roles."""

    def __init__(
        self,
        *,
        participant_ids: tuple[str, ...],
        environment_roles: tuple[GameRole, ...],
    ) -> None:
        """Initialize the mismatch error with both topology sides."""
        super().__init__(
            "Configured participants do not match the environment role topology: "
            f"configured_participant_count={len(participant_ids)} "
            f"participant_ids={participant_ids!r}, "
            f"environment_role_count={len(environment_roles)} "
            f"environment_roles={environment_roles!r}. "
            "Current match scheduling supports exactly one participant for 1-role "
            "environments and exactly two participants for 2-role environments."
        )


class UnsupportedRoleTopologyError(ValueError):
    """Raised when the current match scheduler cannot handle an environment shape."""

    def __init__(self, *, environment_roles: tuple[GameRole, ...]) -> None:
        """Initialize the error with the unsupported environment roles."""
        super().__init__(
            "Current match scheduling supports only 1-role and 2-role environments; "
            f"got environment_roles={environment_roles!r}."
        )


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


@dataclass(frozen=True, slots=True)
class ValidatedMatchPlan:
    """Validated participant/topology/schedule assembly data for one match.

    Match-plan assembly owns topology and schedule validation. Downstream match
    execution code consumes this plan directly without recomputing topology or
    re-validating the schedule.
    """

    participant_ids: tuple[str, ...]
    scheduled_roles: tuple[GameRole, ...]
    schedule: MatchSchedule

    @property
    def participant_count(self) -> int:
        """Return the validated number of configured participants."""
        return len(self.participant_ids)

    @property
    def role_count(self) -> int:
        """Return the validated number of scheduled roles."""
        return len(self.scheduled_roles)

    @property
    def total_games(self) -> int:
        """Return the validated total number of games for the match."""
        return self.schedule.total_games

    @property
    def is_solo(self) -> bool:
        """Whether the validated plan describes a supported 1-role match."""
        return isinstance(self.schedule, SoloMatchSchedule)

    @property
    def is_two_role(self) -> bool:
        """Whether the validated plan describes a supported 2-role match."""
        return isinstance(self.schedule, TwoRoleMatchSchedule)

    @property
    def requires_second_participant(self) -> bool:
        """Whether match execution needs a second configured participant."""
        return self.participant_count == 2

    @property
    def solo_role(self) -> GameRole:
        """Return the only role for a supported solo plan."""
        assert self.is_solo
        return self.scheduled_roles[0]

    @property
    def first_role(self) -> GameRole:
        """Return the first declared role for a supported 2-role plan."""
        assert self.is_two_role
        return self.scheduled_roles[0]

    @property
    def second_role(self) -> GameRole:
        """Return the second declared role for a supported 2-role plan."""
        assert self.is_two_role
        return self.scheduled_roles[1]

    @property
    def two_role_schedule(self) -> TwoRoleMatchSchedule:
        """Return the validated 2-role schedule for a supported 2-role plan."""
        assert isinstance(self.schedule, TwoRoleMatchSchedule)
        return self.schedule

    def role_participant_indexes(
        self, game_number: int
    ) -> tuple[tuple[GameRole, int], ...]:
        """Return the validated role/index schedule for one concrete game."""
        if self.is_solo:
            return ((self.solo_role, 0),)

        participant_indexes: tuple[int, int]
        if (
            game_number
            < self.two_role_schedule.number_of_games_player_one_on_first_role
        ):
            participant_indexes = (0, 1)
        else:
            participant_indexes = (1, 0)
        return (
            (self.first_role, participant_indexes[0]),
            (self.second_role, participant_indexes[1]),
        )


def _validate_supported_match_topology(
    *,
    participant_ids: tuple[str, ...],
    environment_roles: Sequence[GameRole],
) -> tuple[GameRole, ...]:
    """Validate the supported 1-role / 2-role match topologies."""
    role_tuple = tuple(environment_roles)
    if len(role_tuple) not in (1, 2):
        raise UnsupportedRoleTopologyError(environment_roles=role_tuple)
    if len(participant_ids) != len(role_tuple):
        raise ParticipantRoleTopologyMismatchError(
            participant_ids=participant_ids,
            environment_roles=role_tuple,
        )
    return role_tuple


def _validate_schedule_for_supported_topology(
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
    raise UnsupportedRoleTopologyError(environment_roles=role_tuple)


def build_validated_match_plan(
    *,
    participant_ids: Sequence[str],
    environment_roles: Sequence[GameRole],
    schedule: MatchSchedule,
) -> ValidatedMatchPlan:
    """Build one validated plan from participants, environment roles, and schedule.

    This is the single assembly entry point for supported match topologies. It
    preserves the environment role order contract used by the scheduler:
    ``scheduled_roles[0]`` is the first role and ``scheduled_roles[1]`` is the
    second role for supported 2-role environments.
    """
    participant_tuple = tuple(participant_ids)
    scheduled_roles = _validate_supported_match_topology(
        participant_ids=participant_tuple,
        environment_roles=environment_roles,
    )
    validated_schedule = _validate_schedule_for_supported_topology(
        scheduled_roles=scheduled_roles,
        schedule=schedule,
    )
    return ValidatedMatchPlan(
        participant_ids=participant_tuple,
        scheduled_roles=scheduled_roles,
        schedule=validated_schedule,
    )
