"""Integer reduction starting-position args."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from valanga import StateTag

from chipiron.environments.integer_reduction.tags import IntegerReductionStartTag
from chipiron.environments.starting_position import StartingPositionArgs

DEFAULT_INTEGER_REDUCTION_START_VALUE = 15


class IntegerReductionStartingPositionArgsType(StrEnum):
    """Kinds of integer-reduction starting-position args."""

    STANDARD = "integer_reduction_standard"
    VALUE = "integer_reduction_value"


@dataclass(frozen=True)
class IntegerReductionStandardStartingPositionArgs(StartingPositionArgs):
    """Build the canonical integer-reduction starting position."""

    type: Literal[IntegerReductionStartingPositionArgsType.STANDARD] = (
        IntegerReductionStartingPositionArgsType.STANDARD
    )

    def get_start_tag(self) -> StateTag:
        """Return standard integer-reduction start tag."""
        return IntegerReductionStartTag(value=DEFAULT_INTEGER_REDUCTION_START_VALUE)


@dataclass(frozen=True)
class IntegerReductionValueStartingPositionArgs(StartingPositionArgs):
    """Build an integer-reduction starting position from a concrete value."""

    type: Literal[IntegerReductionStartingPositionArgsType.VALUE] = (
        IntegerReductionStartingPositionArgsType.VALUE
    )
    value: int = DEFAULT_INTEGER_REDUCTION_START_VALUE

    def get_start_tag(self) -> StateTag:
        """Return explicit integer-reduction start tag."""
        return IntegerReductionStartTag(value=self.value)
