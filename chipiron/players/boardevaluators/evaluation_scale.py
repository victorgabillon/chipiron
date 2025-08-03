"""Defines the EvaluationScale enum for board evaluation scales."""

from enum import Enum
from typing import Protocol


class EvaluationScale(Enum):
    """
    Enumeration of the evaluation scales used in board evaluations.
    """

    SYMMETRIC_UNIT_INTERVAL = "symmetric_unit_interval"  # [-1.0, 1.0]
    ENTIRE_REAL_AXIS = "entire_real_axis"  # [-inf, +inf] or more pragmatically between a minimum and a maximum value.
    STOCKFISH_BASED = "stockfish_based"  # based on Stockfish evaluation, typically [-1000, 1000] but often normalized to [-1.0, 1.0] or so... (to be checked)


class HasValueOverEnum(Protocol):
    """
    Protocol for enums that provide values for white's evaluation when the node is over.
    """

    VALUE_WHITE_WHEN_OVER_WHITE_WINS: Enum
    VALUE_WHITE_WHEN_OVER_DRAW: Enum
    VALUE_WHITE_WHEN_OVER_BLACK_WINS: Enum


class ValueWhiteWhenOverEntireRealAxis(float, Enum):
    """
    Enum class representing the default values for `value_white` when the node is over.
    """

    VALUE_WHITE_WHEN_OVER_WHITE_WINS = 1000.0
    VALUE_WHITE_WHEN_OVER_DRAW = 0.0
    VALUE_WHITE_WHEN_OVER_BLACK_WINS = -1000.0


class ValueWhiteWhenOverSymmetricUnitInterval(float, Enum):
    """
    Enum class representing the default values for `value_white` when the node is over.
    """

    VALUE_WHITE_WHEN_OVER_WHITE_WINS = 1
    VALUE_WHITE_WHEN_OVER_DRAW = 0.0
    VALUE_WHITE_WHEN_OVER_BLACK_WINS = -1
