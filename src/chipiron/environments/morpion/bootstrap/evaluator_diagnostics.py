"""Persist lightweight Morpion evaluator training diagnostics per generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRows,
    decode_morpion_state_ref_payload,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_regressor_for_inference,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

if TYPE_CHECKING:
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

DIAGNOSTICS_DIRECTORY_NAME = "evaluator_diagnostics"
_WORST_EXAMPLE_LIMIT = 20
_REPRESENTATIVE_WINDOW_SIZE = 10
_EXPECTED_JSON_OBJECT_MAPPING_ERROR = TypeError("Expected a JSON object mapping.")
_EXPECTED_JSON_ARRAY_OF_OBJECTS_ERROR = TypeError("Expected a JSON array of objects.")


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorDiagnosticExample:
    """One representative or high-error supervised row for evaluator debugging."""

    row_index: int
    node_id: str | None
    state_tag: int | None
    depth: int | None
    target_value: float
    prediction_before: float | None
    prediction_after: float | None
    abs_error_before: float | None
    abs_error_after: float | None


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorTrainingDiagnostics:
    """Persisted per-generation evaluator training diagnostics."""

    generation: int
    evaluator_name: str
    dataset_size: int
    created_at: str
    representative_examples: list[MorpionEvaluatorDiagnosticExample]
    worst_examples: list[MorpionEvaluatorDiagnosticExample]
    mae_before: float | None
    mae_after: float | None
    max_abs_error_before: float | None
    max_abs_error_after: float | None


def diagnostics_root_dir(work_dir: str | Path) -> Path:
    """Return the canonical evaluator-diagnostics root for one run directory."""
    return Path(work_dir) / DIAGNOSTICS_DIRECTORY_NAME


def diagnostics_generation_dir(work_dir: str | Path, generation: int) -> Path:
    """Return the per-generation evaluator-diagnostics directory."""
    return diagnostics_root_dir(work_dir) / f"generation_{generation:06d}"


def diagnostics_path(
    work_dir: str | Path,
    generation: int,
    evaluator_name: str,
) -> Path:
    """Return the canonical diagnostics JSON path for one evaluator generation."""
    return diagnostics_generation_dir(work_dir, generation) / f"{evaluator_name}.json"


def diagnostics_history_path(work_dir: str | Path, evaluator_name: str) -> Path:
    """Return the append-only diagnostics history path for one evaluator."""
    return diagnostics_root_dir(work_dir) / f"{evaluator_name}_history.jsonl"


def representative_row_indexes(dataset_size: int) -> tuple[int, ...]:
    """Return deterministic representative row indexes across scale windows."""
    if dataset_size <= 0:
        return ()

    starts = [0, 10]
    window_start = 100
    while window_start < dataset_size:
        starts.append(window_start)
        window_start *= 10

    indexes: list[int] = []
    seen: set[int] = set()
    for start in starts:
        if start >= dataset_size:
            continue
        end = min(start + _REPRESENTATIVE_WINDOW_SIZE, dataset_size)
        for index in range(start, end):
            if index in seen:
                continue
            indexes.append(index)
            seen.add(index)
    return tuple(indexes)


def build_evaluator_training_diagnostics(
    *,
    generation: int,
    evaluator_name: str,
    rows: MorpionSupervisedRows,
    created_at: str,
    model_after: MorpionRegressor,
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    feature_names: tuple[str, ...] = (),
    model_before: MorpionRegressor | None = None,
) -> MorpionEvaluatorTrainingDiagnostics:
    """Build lightweight evaluator diagnostics from existing supervised rows."""
    row_examples, targets = _row_examples_and_targets(
        rows=rows,
        feature_subset_name=feature_subset_name,
        feature_names=feature_names,
    )
    predictions_before = _predict_rows(model_before, row_examples)
    predictions_after = _predict_rows(model_after, row_examples)

    indexed_examples = [
        _example_from_row(
            row_index=row_index,
            row_info=row_info,
            prediction_before=prediction_before,
            prediction_after=prediction_after,
        )
        for row_index, (row_info, prediction_before, prediction_after) in enumerate(
            zip(row_examples, predictions_before, predictions_after, strict=True)
        )
    ]

    representative_examples = [
        indexed_examples[index]
        for index in representative_row_indexes(len(indexed_examples))
    ]
    worst_examples = _worst_examples(indexed_examples)
    abs_errors_before = [
        error
        for error in (example.abs_error_before for example in indexed_examples)
        if error is not None
    ]
    abs_errors_after = [
        error
        for error in (example.abs_error_after for example in indexed_examples)
        if error is not None
    ]

    return MorpionEvaluatorTrainingDiagnostics(
        generation=generation,
        evaluator_name=evaluator_name,
        dataset_size=len(targets),
        created_at=created_at,
        representative_examples=representative_examples,
        worst_examples=worst_examples,
        mae_before=_mean(abs_errors_before),
        mae_after=_mean(abs_errors_after),
        max_abs_error_before=max(abs_errors_before, default=None),
        max_abs_error_after=max(abs_errors_after, default=None),
    )


def save_evaluator_training_diagnostics(
    diagnostics: MorpionEvaluatorTrainingDiagnostics,
    path: str | Path,
) -> None:
    """Persist one evaluator diagnostics artifact as human-readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_evaluator_training_diagnostics_history(
    diagnostics: MorpionEvaluatorTrainingDiagnostics,
    work_dir: str | Path,
) -> None:
    """Append one evaluator diagnostics record to its per-run history."""
    target = diagnostics_history_path(work_dir, diagnostics.evaluator_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(diagnostics), sort_keys=True) + "\n")


def load_evaluator_training_diagnostics(
    path: str | Path,
) -> MorpionEvaluatorTrainingDiagnostics:
    """Load one persisted evaluator diagnostics artifact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = dict(_mapping(raw))
    representative_data = data.get("representative_examples", [])
    worst_data = data.get("worst_examples", [])
    return MorpionEvaluatorTrainingDiagnostics(
        generation=int(data["generation"]),
        evaluator_name=str(data["evaluator_name"]),
        dataset_size=int(data["dataset_size"]),
        created_at=str(data["created_at"]),
        representative_examples=[
            _example_from_dict(item) for item in _list_of_mappings(representative_data)
        ],
        worst_examples=[
            _example_from_dict(item) for item in _list_of_mappings(worst_data)
        ],
        mae_before=_optional_float(data.get("mae_before")),
        mae_after=_optional_float(data.get("mae_after")),
        max_abs_error_before=_optional_float(data.get("max_abs_error_before")),
        max_abs_error_after=_optional_float(data.get("max_abs_error_after")),
    )


def load_latest_evaluator_training_diagnostics(
    work_dir: str | Path,
) -> dict[str, MorpionEvaluatorTrainingDiagnostics]:
    """Load the latest available evaluator diagnostics set for one run."""
    root = diagnostics_root_dir(work_dir)
    if not root.is_dir():
        return {}

    generation_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("generation_")
    )
    if not generation_dirs:
        return {}

    latest_dir = generation_dirs[-1]
    diagnostics_by_evaluator: dict[str, MorpionEvaluatorTrainingDiagnostics] = {}
    for path in sorted(latest_dir.glob("*.json")):
        diagnostics = load_evaluator_training_diagnostics(path)
        diagnostics_by_evaluator[diagnostics.evaluator_name] = diagnostics
    return diagnostics_by_evaluator


def load_previous_evaluator_for_diagnostics(
    model_bundle_path: str | Path | None,
) -> MorpionRegressor | None:
    """Load one previous evaluator bundle for before-training predictions."""
    if model_bundle_path is None:
        return None
    path = Path(model_bundle_path)
    if not path.is_dir():
        return None
    try:
        return load_morpion_regressor_for_inference(path)
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class _PreparedDiagnosticRow:
    """Prepared row fields needed for prediction and JSON diagnostics."""

    node_id: str | None
    state_tag: int | None
    depth: int | None
    target_value: float
    input_tensor: torch.Tensor


def _row_examples_and_targets(
    *,
    rows: MorpionSupervisedRows,
    feature_subset_name: str,
    feature_names: tuple[str, ...],
) -> tuple[list[_PreparedDiagnosticRow], list[float]]:
    """Prepare feature tensors and display metadata for all rows once."""
    subset = resolve_morpion_feature_subset(
        feature_subset_name=feature_subset_name,
        feature_names=feature_names if feature_names else None,
    )
    dynamics = MorpionDynamics()
    converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=subset,
    )
    prepared_rows: list[_PreparedDiagnosticRow] = []
    targets: list[float] = []
    for row in rows.rows:
        chipiron_state = dynamics.wrap_atomheart_state(
            decode_morpion_state_ref_payload(row.state_ref_payload)
        )
        prepared_rows.append(
            _PreparedDiagnosticRow(
                node_id=row.node_id,
                state_tag=_optional_int(getattr(chipiron_state, "tag", None)),
                depth=row.depth,
                target_value=row.target_value,
                input_tensor=converter.state_to_tensor(chipiron_state),
            )
        )
        targets.append(row.target_value)
    return prepared_rows, targets


def _predict_rows(
    model: MorpionRegressor | None,
    rows: list[_PreparedDiagnosticRow],
) -> list[float | None]:
    """Return one prediction per row or ``None`` when no model is available."""
    if model is None:
        return [None] * len(rows)
    if not rows:
        return []

    model.eval()
    stacked_inputs = torch.stack([row.input_tensor for row in rows])
    with torch.no_grad():
        predictions = model(stacked_inputs).squeeze(-1).detach().cpu().tolist()
    if isinstance(predictions, float):
        return [float(predictions)]
    return [float(prediction) for prediction in predictions]


def _example_from_row(
    *,
    row_index: int,
    row_info: _PreparedDiagnosticRow,
    prediction_before: float | None,
    prediction_after: float | None,
) -> MorpionEvaluatorDiagnosticExample:
    """Build one persisted example entry from one prepared row."""
    return MorpionEvaluatorDiagnosticExample(
        row_index=row_index,
        node_id=row_info.node_id,
        state_tag=row_info.state_tag,
        depth=row_info.depth,
        target_value=row_info.target_value,
        prediction_before=prediction_before,
        prediction_after=prediction_after,
        abs_error_before=_abs_error(row_info.target_value, prediction_before),
        abs_error_after=_abs_error(row_info.target_value, prediction_after),
    )


def _worst_examples(
    examples: list[MorpionEvaluatorDiagnosticExample],
) -> list[MorpionEvaluatorDiagnosticExample]:
    """Return the highest after-error rows in descending error order."""
    ranked_examples = [
        example for example in examples if example.abs_error_after is not None
    ]
    ranked_examples.sort(
        key=lambda example: float(example.abs_error_after or 0.0),
        reverse=True,
    )
    return ranked_examples[:_WORST_EXAMPLE_LIMIT]


def _example_from_dict(data: dict[str, Any]) -> MorpionEvaluatorDiagnosticExample:
    """Deserialize one persisted diagnostic example."""
    return MorpionEvaluatorDiagnosticExample(
        row_index=int(data["row_index"]),
        node_id=_optional_str(data.get("node_id")),
        state_tag=_optional_int(data.get("state_tag")),
        depth=_optional_int(data.get("depth")),
        target_value=float(data["target_value"]),
        prediction_before=_optional_float(data.get("prediction_before")),
        prediction_after=_optional_float(data.get("prediction_after")),
        abs_error_before=_optional_float(data.get("abs_error_before")),
        abs_error_after=_optional_float(data.get("abs_error_after")),
    )


def _mapping(value: object) -> dict[str, Any]:
    """Return one string-keyed mapping or raise ``TypeError``."""
    if not isinstance(value, dict):
        raise _EXPECTED_JSON_OBJECT_MAPPING_ERROR
    return dict(value)


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    """Return one list of mapping payloads or raise ``TypeError``."""
    if not isinstance(value, list):
        raise _EXPECTED_JSON_ARRAY_OF_OBJECTS_ERROR
    return [_mapping(item) for item in value]


def _optional_float(value: object) -> float | None:
    """Return one optional float-like value."""
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    """Return one optional int-like value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    """Return one optional string value."""
    if value is None:
        return None
    return str(value)


def _abs_error(target_value: float, prediction: float | None) -> float | None:
    """Return one absolute prediction error or ``None``."""
    if prediction is None:
        return None
    return abs(target_value - prediction)


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or ``None`` for empty lists."""
    if not values:
        return None
    return sum(values) / float(len(values))


__all__ = [
    "MorpionEvaluatorDiagnosticExample",
    "MorpionEvaluatorTrainingDiagnostics",
    "append_evaluator_training_diagnostics_history",
    "build_evaluator_training_diagnostics",
    "diagnostics_generation_dir",
    "diagnostics_history_path",
    "diagnostics_path",
    "diagnostics_root_dir",
    "load_evaluator_training_diagnostics",
    "load_latest_evaluator_training_diagnostics",
    "load_previous_evaluator_for_diagnostics",
    "representative_row_indexes",
    "save_evaluator_training_diagnostics",
]
