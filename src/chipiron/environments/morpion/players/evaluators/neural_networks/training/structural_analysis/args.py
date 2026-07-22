"""Arguments and stable input errors for Morpion structural analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class MorpionStructuralAnalysisArgs:
    """Inputs for one analysis of existing paired evaluator predictions."""

    dataset_file: Path
    comparison_dir: Path
    output_dir: Path
    relational_bundle: Path | None = None
    minimum_ranked_bucket_count: int = 100
    overwrite: bool = False


class MorpionStructuralAnalysisError(RuntimeError):
    """Base class for stable structural-analysis failures."""


class InvalidStructuralAnalysisInputError(MorpionStructuralAnalysisError):
    """Raised when structural-analysis inputs are absent or malformed."""

    @classmethod
    def missing_file(
        cls, label: str, path: Path
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for one required missing file."""
        return cls(f"Missing structural-analysis {label}: {path!s}.")

    @classmethod
    def wrong_schema(cls, actual: object) -> InvalidStructuralAnalysisInputError:
        """Return an error for an incompatible comparison schema."""
        return cls(
            "Expected comparison schema 'morpion_evaluator_comparison_v1', "
            f"got {actual!r}."
        )

    @classmethod
    def wrong_residual_convention(
        cls, actual: object
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for an incompatible residual convention."""
        return cls(
            f"Expected residual convention 'prediction_minus_target', got {actual!r}."
        )

    @classmethod
    def malformed_summary(cls, detail: str) -> InvalidStructuralAnalysisInputError:
        """Return an error for malformed comparison summary content."""
        return cls(f"Malformed comparison summary: {detail}.")

    @classmethod
    def malformed_prediction(
        cls, line_number: int, detail: str
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for one malformed paired-prediction record."""
        return cls(f"Malformed prediction JSONL line {line_number}: {detail}.")

    @classmethod
    def duplicate_prediction(
        cls, row_index: int
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for a duplicate prediction row."""
        return cls(f"Duplicate paired prediction for dataset row {row_index}.")

    @classmethod
    def unexpected_prediction(
        cls, row_index: int
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for a row outside the summary validation set."""
        return cls(f"Unexpected paired prediction dataset row {row_index}.")

    @classmethod
    def missing_predictions(
        cls, row_indices: tuple[int, ...]
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for missing validation predictions."""
        return cls(f"Missing paired predictions for dataset rows {row_indices[:10]!r}.")

    @classmethod
    def non_finite_value(
        cls, row_index: int, label: str
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error for a non-finite target or prediction."""
        return cls(f"Non-finite {label} at dataset row {row_index}.")

    @classmethod
    def target_mismatch(
        cls, row_index: int, source: float, paired: float
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error when source and paired targets disagree."""
        return cls(
            f"Target mismatch at dataset row {row_index}: source={source!r}, "
            f"paired={paired!r}."
        )

    @classmethod
    def source_rows_missing(
        cls, row_indices: tuple[int, ...]
    ) -> InvalidStructuralAnalysisInputError:
        """Return an error when selected source rows cannot be found."""
        return cls(f"Source dataset is missing selected rows {row_indices[:10]!r}.")

    @classmethod
    def evaluator_roles(cls, detail: str) -> InvalidStructuralAnalysisInputError:
        """Return an error for ambiguous ordinary/relational/MLP roles."""
        return cls(f"Could not resolve evaluator roles: {detail}.")

    @classmethod
    def invalid_minimum_count(cls) -> InvalidStructuralAnalysisInputError:
        """Return an error for an invalid bucket-ranking threshold."""
        return cls("minimum_ranked_bucket_count must be a positive integer.")

    @classmethod
    def output_not_empty(cls, path: Path) -> InvalidStructuralAnalysisInputError:
        """Return an error for an unauthorized output replacement."""
        return cls(
            f"Structural-analysis output directory is not empty: {path!s}. "
            "Pass --overwrite to replace it as a complete artifact set."
        )

    @classmethod
    def invalid_output(cls, path: Path) -> InvalidStructuralAnalysisInputError:
        """Return an error for an output path that is not a directory."""
        return cls(f"Structural-analysis output path is not a directory: {path!s}.")


class InvalidRelationBiasTableError(MorpionStructuralAnalysisError):
    """Raised when a relational bundle has an invalid learned bias table."""

    @classmethod
    def not_relational(cls) -> InvalidRelationBiasTableError:
        """Return an error for a non-relational supplied bundle."""
        return cls("The supplied relation-bias bundle is not a relational model.")

    @classmethod
    def wrong_shape(
        cls, expected_rows: int, actual_shape: tuple[int, ...]
    ) -> InvalidRelationBiasTableError:
        """Return an error for an incompatible relation-bias table shape."""
        return cls(
            f"Expected relation-bias table with {expected_rows} rows, got shape "
            f"{actual_shape!r}."
        )

    @classmethod
    def nonzero_padding(cls) -> InvalidRelationBiasTableError:
        """Return an error when relation row zero is not padding-neutral."""
        return cls("Relation-bias table row zero must be numerically zero.")


__all__ = [
    "InvalidRelationBiasTableError",
    "InvalidStructuralAnalysisInputError",
    "MorpionStructuralAnalysisArgs",
    "MorpionStructuralAnalysisError",
]
