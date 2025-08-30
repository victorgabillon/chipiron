"""Defines the EvaluationScale enum for board evaluation scales."""

from enum import Enum
from typing import Type, Union


class EvaluationScale(str, Enum):
    """
    Enumeration of the evaluation scales used in board evaluations.
    """

    SYMMETRIC_UNIT_INTERVAL = "symmetric_unit_interval"  # [-1.0, 1.0]
    ENTIRE_REAL_AXIS = "entire_real_axis"  # [-inf, +inf]
    STOCKFISH_BASED = "stockfish_based"  # based on Stockfish evaluation


class ValueWhiteWhenOverEntireRealAxis(float, Enum):
    """Enum class representing values for `value_white` when the node is over."""

    VALUE_WHITE_WHEN_OVER_WHITE_WINS = 1000.0
    VALUE_WHITE_WHEN_OVER_DRAW = 0.0
    VALUE_WHITE_WHEN_OVER_BLACK_WINS = -1000.0


class ValueWhiteWhenOverSymmetricUnitInterval(float, Enum):
    """Enum class representing values for `value_white` when the node is over."""

    VALUE_WHITE_WHEN_OVER_WHITE_WINS = 1.0
    VALUE_WHITE_WHEN_OVER_DRAW = 0.0
    VALUE_WHITE_WHEN_OVER_BLACK_WINS = -1.0


# Type alias for the value over enums
ValueOverEnum = Union[
    Type[ValueWhiteWhenOverSymmetricUnitInterval],
    Type[ValueWhiteWhenOverEntireRealAxis],
]


def get_value_over_enum(evaluation_scale: EvaluationScale) -> ValueOverEnum:
    """
    Returns the appropriate ValueWhiteWhenOver enum based on the evaluation scale.
    """
    match evaluation_scale:
        case EvaluationScale.SYMMETRIC_UNIT_INTERVAL:
            return ValueWhiteWhenOverSymmetricUnitInterval
        case EvaluationScale.ENTIRE_REAL_AXIS:
            return ValueWhiteWhenOverEntireRealAxis
        case EvaluationScale.STOCKFISH_BASED:
            return ValueWhiteWhenOverEntireRealAxis
        case _:
            raise ValueError(f"Unsupported evaluation scale: {evaluation_scale}")
