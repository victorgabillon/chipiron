"""
This module defines the MoveSelectorArgs protocol for specifying arguments for MoveSelector construction.
"""

from typing import Protocol

from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """ Protocol for arguments for MoveSelector construction"""
    type: MoveSelectorTypes
