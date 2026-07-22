"""Validated paired-prediction and selected-source-row loading."""
# ruff: noqa: TRY003

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_rows_from_path,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    is_morpion_entity_token_model_kind,
)

from .args import InvalidStructuralAnalysisInputError, MorpionStructuralAnalysisArgs

if TYPE_CHECKING:
    from chipiron.environments.morpion.learning import MorpionSupervisedRow


@dataclass(frozen=True, slots=True)
class EvaluatorRoles:
    """Resolved evaluator names for the three PR 5A1 model families."""

    ordinary: str
    relational: str
    mlp: str


@dataclass(frozen=True, slots=True)
class PairedPrediction:
    """One canonical paired target and finite prediction mapping."""

    row_index: int
    target: float
    predictions: dict[str, float]


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    """Validated comparison metadata and predictions in validation order."""

    summary: dict[str, object]
    row_indices: tuple[int, ...]
    evaluator_names: tuple[str, ...]
    roles: EvaluatorRoles
    predictions: tuple[PairedPrediction, ...]


def load_comparison_inputs(args: MorpionStructuralAnalysisArgs) -> ComparisonInputs:
    """Load and fully validate PR 5A1 summary and JSONL artifacts."""
    _validate_paths(args)
    summary_path = args.comparison_dir / "summary.json"
    prediction_path = args.comparison_dir / "predictions.jsonl"
    summary = _load_summary(summary_path)
    row_indices = _summary_row_indices(summary)
    evaluator_names, model_kinds = _summary_evaluators(summary)
    roles = _resolve_roles(model_kinds)
    predictions = _load_predictions(
        prediction_path,
        row_indices=row_indices,
        evaluator_names=evaluator_names,
    )
    _validate_summary_dataset(summary, args.dataset_file)
    return ComparisonInputs(
        summary=summary,
        row_indices=row_indices,
        evaluator_names=evaluator_names,
        roles=roles,
        predictions=predictions,
    )


def load_selected_source_rows(
    dataset_file: Path,
    comparison: ComparisonInputs,
) -> tuple[MorpionSupervisedRow, ...]:
    """Stream once and retain only source rows selected by the comparison."""
    selected = set(comparison.row_indices)
    found: dict[int, MorpionSupervisedRow] = {}
    maximum_index = comparison.row_indices[-1]
    prediction_by_index = {
        prediction.row_index: prediction for prediction in comparison.predictions
    }
    for row_index, row in enumerate(
        iter_morpion_supervised_rows_from_path(
            dataset_file,
            max_rows=maximum_index + 1,
        )
    ):
        if row_index not in selected:
            continue
        paired_target = prediction_by_index[row_index].target
        if not math.isclose(
            row.target_value,
            paired_target,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise InvalidStructuralAnalysisInputError.target_mismatch(
                row_index,
                row.target_value,
                paired_target,
            )
        found[row_index] = row
    missing = tuple(index for index in comparison.row_indices if index not in found)
    if missing:
        raise InvalidStructuralAnalysisInputError.source_rows_missing(missing)
    return tuple(found[index] for index in comparison.row_indices)


def _validate_paths(args: MorpionStructuralAnalysisArgs) -> None:
    """Validate required files and scalar arguments before reading content."""
    if not args.dataset_file.is_file():
        raise _missing_file_error("dataset file", args.dataset_file)
    for file_name in ("summary.json", "predictions.jsonl"):
        path = args.comparison_dir / file_name
        if not path.is_file():
            raise _missing_file_error(file_name, path)
    if args.relational_bundle is not None and not args.relational_bundle.is_dir():
        raise _missing_file_error("relational bundle directory", args.relational_bundle)
    if (
        isinstance(args.minimum_ranked_bucket_count, bool)
        or args.minimum_ranked_bucket_count <= 0
    ):
        raise InvalidStructuralAnalysisInputError.invalid_minimum_count()


def _load_summary(path: Path) -> dict[str, object]:
    """Decode and validate the comparison schema headers."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise _summary_error("top-level value must be a string-keyed mapping")
    summary = cast("dict[str, object]", raw)
    if summary.get("schema") != "morpion_evaluator_comparison_v1":
        raise InvalidStructuralAnalysisInputError.wrong_schema(summary.get("schema"))
    if summary.get("residual_convention") != "prediction_minus_target":
        raise InvalidStructuralAnalysisInputError.wrong_residual_convention(
            summary.get("residual_convention")
        )
    return summary


def _summary_row_indices(summary: dict[str, object]) -> tuple[int, ...]:
    """Return unique strictly increasing canonical validation row indices."""
    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise _summary_error("dataset must be a mapping")
    raw_indices = dataset.get("validation_row_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise _summary_error("validation_row_indices must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_indices
    ):
        raise _summary_error("validation row indices must be integers")
    indices = cast("tuple[int, ...]", tuple(raw_indices))
    if any(current <= previous for previous, current in pairwise(indices)):
        raise _summary_error(
            "validation row indices must be unique and strictly increasing"
        )
    declared_count = dataset.get("validation_row_count")
    if declared_count != len(indices):
        raise _summary_error(
            "validation row count does not match validation_row_indices"
        )
    return indices


def _summary_evaluators(
    summary: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return evaluator names and canonical model-kind metadata."""
    raw_evaluators = summary.get("evaluators")
    if not isinstance(raw_evaluators, Mapping) or not raw_evaluators:
        raise _summary_error("evaluators must be a non-empty mapping")
    model_kinds: dict[str, str] = {}
    for raw_name, raw_metadata in raw_evaluators.items():
        if not isinstance(raw_name, str) or not isinstance(raw_metadata, Mapping):
            raise _summary_error("evaluator entries must be named mappings")
        model_kind = raw_metadata.get("model_kind")
        if not isinstance(model_kind, str):
            raise _summary_error(f"evaluator {raw_name!r} is missing model_kind")
        model_kinds[raw_name] = model_kind
    return tuple(sorted(model_kinds)), model_kinds


def _resolve_roles(model_kinds: dict[str, str]) -> EvaluatorRoles:
    """Resolve exactly one ordinary Transformer, relational Transformer, and MLP."""
    ordinary = tuple(
        name
        for name, kind in model_kinds.items()
        if is_morpion_entity_token_model_kind(kind)
    )
    relational = tuple(
        name
        for name, kind in model_kinds.items()
        if is_relational_entity_token_model_kind(kind)
    )
    mlp = tuple(name for name, kind in model_kinds.items() if kind == "mlp")
    if len(ordinary) != 1 or len(relational) != 1 or len(mlp) != 1:
        raise _roles_error(
            f"ordinary={ordinary!r}, relational={relational!r}, mlp={mlp!r}"
        )
    return EvaluatorRoles(ordinary=ordinary[0], relational=relational[0], mlp=mlp[0])


def _load_predictions(
    path: Path,
    *,
    row_indices: tuple[int, ...],
    evaluator_names: tuple[str, ...],
) -> tuple[PairedPrediction, ...]:
    """Load exactly one finite prediction record for every selected row."""
    expected = set(row_indices)
    loaded: dict[int, PairedPrediction] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidStructuralAnalysisInputError.malformed_prediction(
                    line_number, "invalid JSON"
                ) from exc
            if not isinstance(raw, Mapping):
                raise InvalidStructuralAnalysisInputError.malformed_prediction(
                    line_number, "record must be a mapping"
                )
            row_index = raw.get("row_index")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise InvalidStructuralAnalysisInputError.malformed_prediction(
                    line_number, "row_index must be an integer"
                )
            if row_index not in expected:
                raise InvalidStructuralAnalysisInputError.unexpected_prediction(
                    row_index
                )
            if row_index in loaded:
                raise InvalidStructuralAnalysisInputError.duplicate_prediction(
                    row_index
                )
            target = _finite_number(raw.get("target"), row_index, "target")
            raw_predictions = raw.get("predictions")
            if not isinstance(raw_predictions, Mapping):
                raise InvalidStructuralAnalysisInputError.malformed_prediction(
                    line_number, "predictions must be a mapping"
                )
            if set(raw_predictions) != set(evaluator_names):
                raise InvalidStructuralAnalysisInputError.malformed_prediction(
                    line_number, "prediction evaluator names do not match summary"
                )
            predictions = {
                name: _finite_number(
                    raw_predictions[name], row_index, f"prediction for {name!r}"
                )
                for name in evaluator_names
            }
            loaded[row_index] = PairedPrediction(row_index, target, predictions)
    missing = tuple(index for index in row_indices if index not in loaded)
    if missing:
        raise InvalidStructuralAnalysisInputError.missing_predictions(missing)
    return tuple(loaded[index] for index in row_indices)


def _finite_number(value: object, row_index: int, label: str) -> float:
    """Normalize one finite non-boolean JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidStructuralAnalysisInputError.malformed_prediction(
            row_index, f"{label} must be numeric"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise InvalidStructuralAnalysisInputError.non_finite_value(row_index, label)
    return normalized


def _validate_summary_dataset(summary: dict[str, object], dataset_file: Path) -> None:
    """Require the requested source path to match the comparison metadata."""
    dataset = cast("Mapping[str, object]", summary["dataset"])
    source_path = dataset.get("source_dataset_path")
    if not isinstance(source_path, str):
        raise _summary_error("source_dataset_path must be a string")
    if Path(source_path).resolve() != dataset_file.resolve():
        raise _summary_error("requested dataset path differs from source_dataset_path")


def _missing_file_error(
    label: str,
    path: Path,
) -> InvalidStructuralAnalysisInputError:
    """Return a missing-input error without constructing it at raise sites."""
    return InvalidStructuralAnalysisInputError.missing_file(label, path)


def _summary_error(detail: str) -> InvalidStructuralAnalysisInputError:
    """Return one malformed-summary error."""
    return InvalidStructuralAnalysisInputError.malformed_summary(detail)


def _roles_error(detail: str) -> InvalidStructuralAnalysisInputError:
    """Return one ambiguous evaluator-role error."""
    return InvalidStructuralAnalysisInputError.evaluator_roles(detail)


__all__ = [
    "ComparisonInputs",
    "EvaluatorRoles",
    "PairedPrediction",
    "load_comparison_inputs",
    "load_selected_source_rows",
]
