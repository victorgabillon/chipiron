"""Module for types."""

from enum import StrEnum


# --------- Enums / IDs ---------
class GameKind(StrEnum):
    """Gamekind implementation."""

    CHESS = "chess"
    CHECKERS = "checkers"
