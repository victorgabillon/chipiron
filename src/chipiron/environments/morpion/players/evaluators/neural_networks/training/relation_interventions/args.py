"""Arguments and stable errors for Morpion relation interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class MorpionRelationInterventionError(RuntimeError):
    """Base class for user-facing intervention failures."""


class InvalidMorpionRelationInterventionInputError(MorpionRelationInterventionError):
    """Raised when intervention inputs are inconsistent or malformed."""

    @classmethod
    def invalid(cls, detail: str) -> InvalidMorpionRelationInterventionInputError:
        """Return a stable invalid-input error."""
        return cls(f"Invalid Morpion relation intervention input: {detail}")


class MorpionRelationInterventionInferenceError(MorpionRelationInterventionError):
    """Raised when fixed-model inference violates its contract."""

    @classmethod
    def invalid(cls, detail: str) -> MorpionRelationInterventionInferenceError:
        """Return a stable inference error."""
        return cls(f"Morpion relation intervention inference failed: {detail}")


@dataclass(frozen=True, slots=True)
class MorpionRelationInterventionBundle:
    """One relational seed and its optional matched ordinary context model."""

    seed: int
    relational_bundle: Path
    ordinary_bundle: Path | None = None


@dataclass(frozen=True, slots=True)
class MorpionRelationInterventionArgs:
    """Configuration for one multi-seed fixed-model intervention run."""

    dataset_file: Path
    bundles: tuple[MorpionRelationInterventionBundle, ...]
    output_dir: Path
    structural_analysis_dir: Path | None = None
    max_rows: int | None = 50_000
    validation_fraction: float = 0.2
    batch_size: int = 64
    device: str = "auto"
    overwrite: bool = False


__all__ = [
    "InvalidMorpionRelationInterventionInputError",
    "MorpionRelationInterventionArgs",
    "MorpionRelationInterventionBundle",
    "MorpionRelationInterventionError",
    "MorpionRelationInterventionInferenceError",
]
