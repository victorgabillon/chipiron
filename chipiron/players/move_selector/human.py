from dataclasses import dataclass
from typing import Literal

from .move_selector_types import MoveSelectorTypes


@dataclass
class CommandLineHumanPlayerArgs:
    type: Literal[MoveSelectorTypes.CommandLineHuman]  # for serialization
