"""Target scaling for Morpion regressors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class InvalidMorpionTargetTransformError(ValueError):
    """Raised when Morpion target-transform metadata is invalid."""

    @classmethod
    def non_finite_mean(cls) -> InvalidMorpionTargetTransformError:
        """Return the non-finite-mean error."""
        return cls("Morpion target-transform mean must be finite.")

    @classmethod
    def invalid_standard_deviation(cls) -> InvalidMorpionTargetTransformError:
        """Return the invalid-standard-deviation error."""
        return cls(
            "Enabled Morpion target-transform standard deviation must be finite "
            "and strictly positive."
        )


@dataclass(frozen=True, slots=True)
class MorpionTargetTransform:
    """Affine transform between original and standardized Morpion targets."""

    enabled: bool = False
    mean: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        """Validate persisted transform values."""
        if not math.isfinite(self.mean):
            raise InvalidMorpionTargetTransformError.non_finite_mean()
        if self.enabled and (
            not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0.0
        ):
            raise InvalidMorpionTargetTransformError.invalid_standard_deviation()

    def normalize(self, value: Tensor) -> Tensor:
        """Return targets in the internal training scale."""
        if not self.enabled:
            return value
        return (value - self.mean) / self.standard_deviation

    def denormalize(self, value: Tensor) -> Tensor:
        """Return predictions in the public original target scale."""
        if not self.enabled:
            return value
        return self.mean + self.standard_deviation * value


__all__ = [
    "InvalidMorpionTargetTransformError",
    "MorpionTargetTransform",
]
