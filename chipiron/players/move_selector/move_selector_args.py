from typing import Protocol

from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """ Protocol for arguments for MoveSelector construction"""
    type: MoveSelectorTypes
