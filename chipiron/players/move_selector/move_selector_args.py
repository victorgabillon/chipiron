from .move_selector_types import MoveSelectorTypes
from dataclasses import dataclass
from typing import Protocol


@dataclass
class MoveSelectorArgs(Protocol):
    """ Protocol for arguments for MoveSelector construction"""
    type: MoveSelectorTypes
