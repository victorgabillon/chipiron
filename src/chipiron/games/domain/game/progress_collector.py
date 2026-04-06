"""Module collecting the progress of computing moves by each player."""

from dataclasses import dataclass, field
from typing import Protocol

from chipiron.core.roles import GameRole, MutableRoleAssignment
from chipiron.displays.gui_protocol import UpdParticipantProgress
from chipiron.displays.gui_publisher import GuiPublisher


class PlayerProgressCollectorP(Protocol):
    """Object defining the protocol for setting progress values by role."""

    def progress(self, role: GameRole, value: int | None) -> None:
        """Store or publish progress for one role."""
        ...


def make_progress_by_role() -> MutableRoleAssignment[int | None]:
    """Create the mutable store used for per-role progress values."""
    return {}


@dataclass(slots=True)
class PlayerProgressCollector:
    """Collect the progress of move computation by game role."""

    progress_by_role: MutableRoleAssignment[int | None] = field(
        default_factory=make_progress_by_role
    )

    def progress(self, role: GameRole, value: int | None) -> None:
        """Store progress for the given role."""
        self.progress_by_role[role] = value

    def progress_for_role(self, role: GameRole) -> int | None:
        """Return the latest known progress for the given role."""
        return self.progress_by_role.get(role)


def make_publishers() -> list[GuiPublisher]:
    """Create publishers."""
    return []


@dataclass(slots=True)
class PlayerProgressCollectorObservable(PlayerProgressCollectorP):
    """Collects progress and publishes GUI payloads."""

    publishers: list[GuiPublisher] = field(default_factory=make_publishers)
    progress_collector: PlayerProgressCollector = field(
        default_factory=PlayerProgressCollector
    )

    def progress(self, role: GameRole, value: int | None) -> None:
        """Store and publish progress for the given role."""
        self.progress_collector.progress(role, value)
        self._publish(role=role, value=value)

    def _publish(self, role: GameRole, value: int | None) -> None:
        """Publish a progress update to all GUI publishers."""
        payload = UpdParticipantProgress(
            role=role,
            progress_percent=value,
        )
        for pub in self.publishers:
            pub.publish(payload)
