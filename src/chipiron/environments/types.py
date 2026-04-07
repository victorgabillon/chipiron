"""Module for types."""

from enum import StrEnum


# --------- Enums / IDs ---------
class GameKind(StrEnum):
    """Gamekind implementation."""

    CHESS = "chess"
    CHECKERS = "checkers"
    INTEGER_REDUCTION = "integer_reduction"
    MORPION = "morpion"
