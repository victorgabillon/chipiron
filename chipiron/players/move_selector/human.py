from typing import Literal
from dataclasses import dataclass
from .move_selector_types import MoveSelectorTypes


@dataclass
class HumanPlayerArgs:
    type: Literal[MoveSelectorTypes.Human]  # for serialization
