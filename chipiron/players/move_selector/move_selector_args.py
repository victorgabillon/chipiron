"""
This module defines the MoveSelectorArgs protocol for specifying arguments for MoveSelector construction.
"""

from typing import Protocol

from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """Protocol for arguments for MoveSelector construction"""

    type: MoveSelectorTypes

    def is_human(self) -> bool:
        return self.type.is_human()
