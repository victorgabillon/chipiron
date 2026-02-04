"""Module for starting position."""
from typing import Protocol

from valanga import StateTag


class StartingPositionArgs(Protocol):
    """Protocol for objects that provide a starting state tag."""

    def get_start_tag(self) -> StateTag:
        """Return the starting state tag."""
        ...
