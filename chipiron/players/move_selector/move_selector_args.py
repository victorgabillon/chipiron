from .move_selector_types import MoveSelectorTypes
from dataclasses import dataclass


@dataclass
class MoveSelectorArgs:
    """ Base class for arguments for MoveSelector construction"""
    type: MoveSelectorTypes
