"""Integer reduction start-tag representations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegerReductionStartTag:
    """Lossless integer-reduction starting tag."""

    value: int
