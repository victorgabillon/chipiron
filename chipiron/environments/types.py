"""Module for types."""
from enum import Enum


# --------- Enums / IDs ---------
class GameKind(str, Enum):
    """Gamekind implementation."""
    CHESS = "chess"
    CHECKERS = "checkers"
