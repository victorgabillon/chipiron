"""Module for starting position."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from valanga import StateTag


class StartingPositionArgs(Protocol):
    """Protocol for objects that provide a starting state tag."""

    def get_start_tag(self) -> StateTag:
        """Return the starting state tag."""
        ...
