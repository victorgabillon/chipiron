"""Args dataclass for the Stockfish move selector."""

from dataclasses import dataclass
from typing import Literal

from .move_selector_types import MoveSelectorTypes


@dataclass(frozen=True, slots=True)
class StockfishSelectorArgs:
    """Serializable arguments for Stockfish selector construction."""

    type: Literal[MoveSelectorTypes.STOCKFISH] = MoveSelectorTypes.STOCKFISH
    depth: int = 20
    time_limit: float = 0.1
