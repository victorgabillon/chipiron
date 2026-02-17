"""Args dataclass for the random move selector."""

from dataclasses import dataclass
from typing import Literal

from .move_selector_types import MoveSelectorTypes


@dataclass(frozen=True, slots=True)
class RandomSelectorArgs:
    """Serializable arguments for the random move selector."""

    type: Literal[MoveSelectorTypes.RANDOM] = MoveSelectorTypes.RANDOM
