"""Role-aware game result reporting structures."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from chipiron.core.roles import GameRole, ParticipantId, format_game_role

if TYPE_CHECKING:
    from valanga import StateTag
    from valanga.game import ActionName
else:
    ActionName = str
    StateTag = str


class FinalGameResult(StrEnum):
    """Enum representing the final result of a game."""

    WIN_FOR_WHITE = "win_for_white"
    WIN_FOR_BLACK = "win_for_black"
    DRAW = "draw"


class RoleOutcome(StrEnum):
    """Outcome for a single structural game role."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    UNKNOWN = "unknown"


def make_serializable_participant_mapping() -> dict[str, ParticipantId]:
    """Build the default participant mapping for serialized reports."""
    return {}


def make_serializable_role_outcomes() -> dict[str, RoleOutcome]:
    """Build the default role-outcome mapping for serialized reports."""
    return {}


def make_serializable_winner_roles() -> list[str]:
    """Build the default winner-role list for serialized reports."""
    return []


@dataclass(frozen=True, slots=True)
class RoleGameResult:
    """Role-first result view used by reporting and aggregation layers."""

    result_by_role: dict[GameRole, RoleOutcome]
    winner_roles: tuple[GameRole, ...] = ()
    reason: str | None = None

    def to_serializable_result_by_role(self) -> dict[str, RoleOutcome]:
        """Serialize the role-keyed result map with readable role labels."""
        return {
            format_game_role(role): outcome
            for role, outcome in self.result_by_role.items()
        }

    def to_serializable_winner_roles(self) -> list[str]:
        """Serialize winner roles for YAML/reporting consumers."""
        return [format_game_role(role) for role in self.winner_roles]


@dataclass
class GameReport:
    """Dataclass representing a game report."""

    final_game_result: FinalGameResult
    action_history: list[ActionName]
    state_tag_history: list[StateTag]
    participant_id_by_role: dict[str, ParticipantId] = field(
        default_factory=make_serializable_participant_mapping
    )
    result_by_role: dict[str, RoleOutcome] = field(
        default_factory=make_serializable_role_outcomes
    )
    winner_roles: list[str] = field(default_factory=make_serializable_winner_roles)
    result_reason: str | None = None

    @property
    def result_by_participant(self) -> dict[ParticipantId, RoleOutcome]:
        """Return participant results derived from the role assignment."""
        return {
            participant_id: self.result_by_role[role_label]
            for role_label, participant_id in self.participant_id_by_role.items()
            if role_label in self.result_by_role
        }
