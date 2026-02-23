"""Checkers-specific type aliases backed by atomheart."""

from atomheart.games.checkers import CheckersDynamics, CheckersRules, CheckersState
from atomheart.games.checkers.move import MoveKey as CheckersMoveKey
from atomheart.games.checkers.move import move_name as checkers_move_name

__all__ = [
    "CheckersDynamics",
    "CheckersMoveKey",
    "CheckersRules",
    "CheckersState",
    "checkers_move_name",
]
