"""
This module provides board evaluators for the chipiron game.

The board evaluators are used to evaluate the current state of the game board and assign a value to it.
"""

from .board_evaluator import BoardEvaluator, ValueWhiteWhenOver

__all__ = ["BoardEvaluator", "ValueWhiteWhenOver"]
