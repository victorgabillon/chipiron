"""Durable artifact-contract helpers for the Morpion bootstrap pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

MorpionPipelineDatasetStatus = Literal[
    "not_started",
    "exporting_tree",
    "extracting_rows",
    "done",
    "failed",
]

MorpionPipelineTrainingStatus = Literal[
    "not_started",
    "training",
    "selecting",
    "done",
    "failed",
]

MorpionPipelineStageName = Literal["dataset", "training"]

_DATASET_STATUSES: frozenset[str] = frozenset(
    {
        "not_started",
        "exporting_tree",
        "extracting_rows",
        "done",
        "failed",
    }
)
_TRAINING_STATUSES: frozenset[str] = frozenset(
    {"not_started", "training", "selecting", "done", "failed"}
)
_STAGE_NAMES: frozenset[str] = frozenset({"dataset", "training"})


def _empty_metadata() -> dict[str, object]:
    """Return a typed empty metadata mapping."""
    return {}


def _empty_model_bundle_paths() -> dict[str, str]:
    """Return a typed empty model bundle mapping."""
    return {}


def _empty_evaluator_results() -> dict[str, MorpionPipelineEvaluatorTrainingResult]:
    """Return a typed empty evaluator training-result mapping."""
    return {}


class InvalidMorpionPipelineArtifactError(ValueError):
    """Raised when one persisted pipeline artifact payload is malformed."""


class MissingMorpionPipelineArtifactError(FileNotFoundError):
    """Raised when one required pipeline artifact file does not exist."""


def _invalid_payload_error() -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-payload error for top-level artifact payloads."""
    return InvalidMorpionPipelineArtifactError(
        "Morpion pipeline artifact payload must be a mapping with string keys."
    )


def _missing_manifest_error(path: Path) -> MissingMorpionPipelineArtifactError:
    """Return the stable missing-manifest error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion pipeline manifest does not exist: {path}"
    )


def _invalid_manifest_json_error(path: Path) -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-manifest-json error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion pipeline manifest at {path} is not valid JSON."
    )


def _missing_active_model_error(path: Path) -> MissingMorpionPipelineArtifactError:
    """Return the stable missing-active-model error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion pipeline active-model artifact does not exist: {path}"
    )


def _invalid_active_model_json_error(path: Path) -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-active-model-json error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion pipeline active-model artifact at {path} is not valid JSON."
    )


def _invalid_field_error(
    field_name: str,
    detail: str,
) -> InvalidMorpionPipelineArtifactError:
    """Return a stable invalid-field error for pipeline artifact payloads."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion pipeline artifact field `{field_name}` {detail}."
    )


def _require_generation(value: object, *, field_name: str) -> int:
    """Return one validated non-negative generation index."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_field_error(field_name, "must be an integer")
    if value < 0:
        raise _invalid_field_error(field_name, "must be >= 0")
    return value


def _require_str(value: object, *, field_name: str) -> str:
    """Return one required string field."""
    if not isinstance(value, str):
        raise _invalid_field_error(field_name, "must be a string")
    return value


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    """Return one required non-empty string field."""
    normalized = _require_str(value, field_name=field_name)
    if normalized == "":
        raise _invalid_field_error(field_name, "must be a non-empty string")
    return normalized


def _optional_path_str(value: object, *, field_name: str) -> str | None:
    """Return one optional persisted-path field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_field_error(field_name, "must be a string or null")
    return value


def _optional_str(value: object, *, field_name: str) -> str | None:
    """Return one optional string field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_field_error(field_name, "must be a string or null")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    """Return one validated non-negative integer field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_field_error(field_name, "must be a non-negative integer")
    if value < 0:
        raise _invalid_field_error(field_name, "must be a non-negative integer")
    return value


def _optional_generation(value: object, *, field_name: str) -> int | None:
    """Return one optional non-negative generation index."""
    if value is None:
        return None
    return _require_generation(value, field_name=field_name)


def _float_value(value: object, *, field_name: str) -> float:
    """Return one validated finite float field."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid_field_error(field_name, "must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise _invalid_field_error(field_name, "must be a finite number")
    return normalized


def _optional_float_value(value: object, *, field_name: str) -> float | None:
    """Return one optional finite float field."""
    if value is None:
        return None
    return _float_value(value, field_name=field_name)


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    """Return one optional boolean field."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _invalid_field_error(field_name, "must be a boolean or null")
    return value


def _metadata_dict(value: object, *, field_name: str = "metadata") -> dict[str, object]:
    """Return one shallow-copied metadata mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _invalid_field_error(field_name, "must be a mapping")
    raw_mapping = cast("Mapping[object, object]", value)
    if not all(isinstance(key, str) for key in raw_mapping):
        raise _invalid_field_error(field_name, "must use string keys")
    return dict(cast("Mapping[str, object]", raw_mapping))


def _model_bundle_paths_dict(value: object) -> dict[str, str]:
    """Return one shallow-copied evaluator bundle-path mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _invalid_field_error("model_bundle_paths", "must be a mapping")
    raw_mapping = cast("Mapping[object, object]", value)
    copied: dict[str, str] = {}
    for key, raw_path in raw_mapping.items():
        if not isinstance(key, str):
            raise _invalid_field_error(
                "model_bundle_paths", "must use string evaluator names"
            )
        if not isinstance(raw_path, str):
            raise _invalid_field_error(
                "model_bundle_paths", "must use string bundle paths"
            )
        copied[key] = raw_path
    return copied


def _dataset_status(value: object) -> MorpionPipelineDatasetStatus:
    """Return one validated dataset stage status."""
    if not isinstance(value, str) or value not in _DATASET_STATUSES:
        raise _invalid_field_error(
            "dataset_status",
            f"must be one of {sorted(_DATASET_STATUSES)!r}",
        )
    return cast("MorpionPipelineDatasetStatus", value)


def _training_status(value: object) -> MorpionPipelineTrainingStatus:
    """Return one validated training stage status."""
    if not isinstance(value, str) or value not in _TRAINING_STATUSES:
        raise _invalid_field_error(
            "training_status",
            f"must be one of {sorted(_TRAINING_STATUSES)!r}",
        )
    return cast("MorpionPipelineTrainingStatus", value)


def _stage_name(value: object) -> MorpionPipelineStageName:
    """Return one validated pipeline stage name."""
    if not isinstance(value, str) or value not in _STAGE_NAMES:
        raise _invalid_field_error(
            "stage",
            f"must be one of {sorted(_STAGE_NAMES)!r}",
        )
    return cast("MorpionPipelineStageName", value)


def _top_level_mapping(data: object) -> Mapping[str, object]:
    """Return one top-level artifact payload mapping."""
    if not isinstance(data, Mapping):
        raise _invalid_payload_error()
    raw_mapping = cast("Mapping[object, object]", data)
    if not all(isinstance(key, str) for key in raw_mapping):
        raise _invalid_payload_error()
    return cast("Mapping[str, object]", raw_mapping)


@dataclass(frozen=True, slots=True)
class MorpionPipelineGenerationManifest:
    """Immutable manifest describing persisted artifacts for one generation."""

    generation: int
    created_at_utc: str
    runtime_checkpoint_path: str | None = None
    tree_snapshot_path: str | None = None
    rows_path: str | None = None
    model_bundle_paths: dict[str, str] = field(default_factory=_empty_model_bundle_paths)
    selected_evaluator_name: str | None = None
    dataset_status: MorpionPipelineDatasetStatus = "not_started"
    training_status: MorpionPipelineTrainingStatus = "not_started"
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize manifest fields eagerly."""
        object.__setattr__(
            self,
            "generation",
            _require_generation(self.generation, field_name="generation"),
        )
        object.__setattr__(
            self,
            "created_at_utc",
            _require_str(self.created_at_utc, field_name="created_at_utc"),
        )
        object.__setattr__(
            self,
            "runtime_checkpoint_path",
            _optional_path_str(
                self.runtime_checkpoint_path,
                field_name="runtime_checkpoint_path",
            ),
        )
        object.__setattr__(
            self,
            "tree_snapshot_path",
            _optional_path_str(self.tree_snapshot_path, field_name="tree_snapshot_path"),
        )
        object.__setattr__(
            self,
            "rows_path",
            _optional_path_str(self.rows_path, field_name="rows_path"),
        )
        object.__setattr__(
            self,
            "model_bundle_paths",
            _model_bundle_paths_dict(self.model_bundle_paths),
        )
        if self.selected_evaluator_name is not None and not isinstance(
            self.selected_evaluator_name, str
        ):
            raise _invalid_field_error(
                "selected_evaluator_name", "must be a string or null"
            )
        object.__setattr__(self, "dataset_status", _dataset_status(self.dataset_status))
        object.__setattr__(
            self,
            "training_status",
            _training_status(self.training_status),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True, slots=True)
class MorpionPipelineActiveModel:
    """Immutable pointer to the currently selected pipeline model bundle."""

    generation: int
    evaluator_name: str
    model_bundle_path: str
    updated_at_utc: str
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize active-model fields eagerly."""
        object.__setattr__(
            self,
            "generation",
            _require_generation(self.generation, field_name="generation"),
        )
        object.__setattr__(
            self,
            "evaluator_name",
            _require_str(self.evaluator_name, field_name="evaluator_name"),
        )
        object.__setattr__(
            self,
            "model_bundle_path",
            _require_str(self.model_bundle_path, field_name="model_bundle_path"),
        )
        object.__setattr__(
            self,
            "updated_at_utc",
            _require_str(self.updated_at_utc, field_name="updated_at_utc"),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True, slots=True)
class MorpionPipelineEvaluatorTrainingResult:
    """Immutable per-evaluator training result persisted for dashboard consumption."""

    final_loss: float
    elapsed_s: float
    model_bundle_path: str

    def __post_init__(self) -> None:
        """Validate and normalize per-evaluator training result fields eagerly."""
        object.__setattr__(
            self,
            "final_loss",
            _float_value(self.final_loss, field_name="final_loss"),
        )
        object.__setattr__(
            self,
            "elapsed_s",
            _float_value(self.elapsed_s, field_name="elapsed_s"),
        )
        object.__setattr__(
            self,
            "model_bundle_path",
            _require_non_empty_str(
                self.model_bundle_path,
                field_name="model_bundle_path",
            ),
        )


@dataclass(frozen=True, slots=True)
class MorpionPipelineTrainingStatusArtifact:
    """Immutable persisted training-status artifact for one pipeline generation."""

    generation: int
    status: MorpionPipelineTrainingStatus
    updated_at_utc: str
    selected_evaluator_name: str | None = None
    selection_policy: str | None = None
    evaluator_results: dict[str, MorpionPipelineEvaluatorTrainingResult] = field(
        default_factory=_empty_evaluator_results
    )
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize training-status artifact fields eagerly."""
        object.__setattr__(
            self,
            "generation",
            _require_generation(self.generation, field_name="generation"),
        )
        object.__setattr__(self, "status", _training_status(self.status))
        object.__setattr__(
            self,
            "updated_at_utc",
            _require_non_empty_str(
                self.updated_at_utc,
                field_name="updated_at_utc",
            ),
        )
        object.__setattr__(
            self,
            "selected_evaluator_name",
            _optional_str(
                self.selected_evaluator_name,
                field_name="selected_evaluator_name",
            ),
        )
        object.__setattr__(
            self,
            "selection_policy",
            _optional_str(self.selection_policy, field_name="selection_policy"),
        )
        object.__setattr__(
            self,
            "evaluator_results",
            _training_result_mapping(self.evaluator_results),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


def _training_result_mapping(
    value: object,
) -> dict[str, MorpionPipelineEvaluatorTrainingResult]:
    """Return one shallow-copied evaluator training-result mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _invalid_field_error("evaluator_results", "must be a mapping")
    raw_mapping = cast("Mapping[object, object]", value)
    copied: dict[str, MorpionPipelineEvaluatorTrainingResult] = {}
    for key, raw_result in raw_mapping.items():
        if not isinstance(key, str):
            raise _invalid_field_error(
                "evaluator_results", "must use string evaluator names"
            )
        if isinstance(raw_result, MorpionPipelineEvaluatorTrainingResult):
            copied[key] = raw_result
            continue
        copied[key] = pipeline_evaluator_training_result_from_dict(raw_result)
    return copied


@dataclass(frozen=True, slots=True)
class MorpionPipelineStageClaim:
    """Immutable temporary ownership record for one pipeline stage."""

    generation: int
    stage: MorpionPipelineStageName
    claim_id: str
    claimed_at_utc: str
    expires_at_utc: str
    owner: str | None = None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize stage-claim fields eagerly."""
        object.__setattr__(
            self,
            "generation",
            _require_generation(self.generation, field_name="generation"),
        )
        object.__setattr__(self, "stage", _stage_name(self.stage))
        object.__setattr__(
            self,
            "claim_id",
            _require_str(self.claim_id, field_name="claim_id"),
        )
        object.__setattr__(
            self,
            "claimed_at_utc",
            _require_str(self.claimed_at_utc, field_name="claimed_at_utc"),
        )
        object.__setattr__(
            self,
            "expires_at_utc",
            _require_str(self.expires_at_utc, field_name="expires_at_utc"),
        )
        object.__setattr__(
            self,
            "owner",
            _optional_str(self.owner, field_name="owner"),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True, slots=True)
class MorpionReevaluationPatchRow:
    """Immutable bounded node-value update for reevaluation patch artifacts."""

    node_id: str
    direct_value: float
    backed_up_value: float | None = None
    is_exact: bool | None = None
    is_terminal: bool | None = None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize reevaluation-patch row fields eagerly."""
        object.__setattr__(
            self,
            "node_id",
            _require_non_empty_str(self.node_id, field_name="node_id"),
        )
        object.__setattr__(
            self,
            "direct_value",
            _float_value(self.direct_value, field_name="direct_value"),
        )
        object.__setattr__(
            self,
            "backed_up_value",
            _optional_float_value(
                self.backed_up_value,
                field_name="backed_up_value",
            ),
        )
        object.__setattr__(
            self,
            "is_exact",
            _optional_bool(self.is_exact, field_name="is_exact"),
        )
        object.__setattr__(
            self,
            "is_terminal",
            _optional_bool(self.is_terminal, field_name="is_terminal"),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


def _reevaluation_patch_rows_tuple(
    value: object,
) -> tuple[MorpionReevaluationPatchRow, ...]:
    """Return one validated immutable reevaluation-patch row sequence."""
    if not isinstance(value, list | tuple):
        raise _invalid_field_error("rows", "must be a list or tuple of patch rows")
    return tuple(
        item
        if isinstance(item, MorpionReevaluationPatchRow)
        else reevaluation_patch_row_from_dict(item)
        for item in value
    )


@dataclass(frozen=True, slots=True)
class MorpionReevaluationPatch:
    """Immutable reevaluation patch artifact for future tree-value updates."""

    patch_id: str
    created_at_utc: str
    evaluator_generation: int
    evaluator_name: str
    model_bundle_path: str
    rows: tuple[MorpionReevaluationPatchRow, ...]
    tree_generation: int | None = None
    start_cursor: str | None = None
    end_cursor: str | None = None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize reevaluation-patch fields eagerly."""
        object.__setattr__(
            self,
            "patch_id",
            _require_non_empty_str(self.patch_id, field_name="patch_id"),
        )
        object.__setattr__(
            self,
            "created_at_utc",
            _require_non_empty_str(
                self.created_at_utc,
                field_name="created_at_utc",
            ),
        )
        object.__setattr__(
            self,
            "evaluator_generation",
            _require_generation(
                self.evaluator_generation,
                field_name="evaluator_generation",
            ),
        )
        object.__setattr__(
            self,
            "evaluator_name",
            _require_non_empty_str(
                self.evaluator_name,
                field_name="evaluator_name",
            ),
        )
        object.__setattr__(
            self,
            "model_bundle_path",
            _require_non_empty_str(
                self.model_bundle_path,
                field_name="model_bundle_path",
            ),
        )
        object.__setattr__(self, "rows", _reevaluation_patch_rows_tuple(self.rows))
        object.__setattr__(
            self,
            "tree_generation",
            _optional_generation(self.tree_generation, field_name="tree_generation"),
        )
        object.__setattr__(
            self,
            "start_cursor",
            _optional_str(self.start_cursor, field_name="start_cursor"),
        )
        object.__setattr__(
            self,
            "end_cursor",
            _optional_str(self.end_cursor, field_name="end_cursor"),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True, slots=True)
class MorpionReevaluationCursor:
    """Immutable reevaluation cursor artifact tracking bounded patch progress."""

    evaluator_generation: int
    evaluator_name: str
    model_bundle_path: str
    next_node_cursor: str | None
    updated_at_utc: str
    tree_generation: int | None = None
    completed_full_pass_count: int = 0
    last_patch_id: str | None = None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate and normalize reevaluation-cursor fields eagerly."""
        object.__setattr__(
            self,
            "evaluator_generation",
            _require_generation(
                self.evaluator_generation,
                field_name="evaluator_generation",
            ),
        )
        object.__setattr__(
            self,
            "evaluator_name",
            _require_non_empty_str(
                self.evaluator_name,
                field_name="evaluator_name",
            ),
        )
        object.__setattr__(
            self,
            "model_bundle_path",
            _require_non_empty_str(
                self.model_bundle_path,
                field_name="model_bundle_path",
            ),
        )
        object.__setattr__(
            self,
            "next_node_cursor",
            _optional_str(self.next_node_cursor, field_name="next_node_cursor"),
        )
        object.__setattr__(
            self,
            "updated_at_utc",
            _require_non_empty_str(
                self.updated_at_utc,
                field_name="updated_at_utc",
            ),
        )
        object.__setattr__(
            self,
            "tree_generation",
            _optional_generation(self.tree_generation, field_name="tree_generation"),
        )
        object.__setattr__(
            self,
            "completed_full_pass_count",
            _non_negative_int(
                self.completed_full_pass_count,
                field_name="completed_full_pass_count",
            ),
        )
        object.__setattr__(
            self,
            "last_patch_id",
            _optional_str(self.last_patch_id, field_name="last_patch_id"),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


def pipeline_manifest_to_dict(
    manifest: MorpionPipelineGenerationManifest,
) -> dict[str, object]:
    """Serialize one pipeline generation manifest into JSON-friendly data."""
    return {
        "created_at_utc": manifest.created_at_utc,
        "dataset_status": manifest.dataset_status,
        "generation": manifest.generation,
        "metadata": dict(manifest.metadata),
        "model_bundle_paths": dict(manifest.model_bundle_paths),
        "rows_path": manifest.rows_path,
        "runtime_checkpoint_path": manifest.runtime_checkpoint_path,
        "selected_evaluator_name": manifest.selected_evaluator_name,
        "training_status": manifest.training_status,
        "tree_snapshot_path": manifest.tree_snapshot_path,
    }


def pipeline_manifest_from_dict(data: object) -> MorpionPipelineGenerationManifest:
    """Deserialize one pipeline generation manifest from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionPipelineGenerationManifest(
        generation=_require_generation(payload.get("generation"), field_name="generation"),
        created_at_utc=_require_str(
            payload.get("created_at_utc"), field_name="created_at_utc"
        ),
        runtime_checkpoint_path=_optional_path_str(
            payload.get("runtime_checkpoint_path"),
            field_name="runtime_checkpoint_path",
        ),
        tree_snapshot_path=_optional_path_str(
            payload.get("tree_snapshot_path"),
            field_name="tree_snapshot_path",
        ),
        rows_path=_optional_path_str(payload.get("rows_path"), field_name="rows_path"),
        model_bundle_paths=_model_bundle_paths_dict(payload.get("model_bundle_paths")),
        selected_evaluator_name=_optional_path_str(
            payload.get("selected_evaluator_name"),
            field_name="selected_evaluator_name",
        ),
        dataset_status=_dataset_status(payload.get("dataset_status", "not_started")),
        training_status=_training_status(
            payload.get("training_status", "not_started")
        ),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def pipeline_active_model_to_dict(
    active_model: MorpionPipelineActiveModel,
) -> dict[str, object]:
    """Serialize one active-model record into JSON-friendly data."""
    return {
        "evaluator_name": active_model.evaluator_name,
        "generation": active_model.generation,
        "metadata": dict(active_model.metadata),
        "model_bundle_path": active_model.model_bundle_path,
        "updated_at_utc": active_model.updated_at_utc,
    }


def pipeline_active_model_from_dict(data: object) -> MorpionPipelineActiveModel:
    """Deserialize one active-model record from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionPipelineActiveModel(
        generation=_require_generation(payload.get("generation"), field_name="generation"),
        evaluator_name=_require_str(
            payload.get("evaluator_name"), field_name="evaluator_name"
        ),
        model_bundle_path=_require_str(
            payload.get("model_bundle_path"), field_name="model_bundle_path"
        ),
        updated_at_utc=_require_str(
            payload.get("updated_at_utc"), field_name="updated_at_utc"
        ),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def pipeline_evaluator_training_result_to_dict(
    result: MorpionPipelineEvaluatorTrainingResult,
) -> dict[str, object]:
    """Serialize one per-evaluator training result into JSON-friendly data."""
    return {
        "elapsed_s": result.elapsed_s,
        "final_loss": result.final_loss,
        "model_bundle_path": result.model_bundle_path,
    }


def pipeline_evaluator_training_result_from_dict(
    data: object,
) -> MorpionPipelineEvaluatorTrainingResult:
    """Deserialize one per-evaluator training result from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionPipelineEvaluatorTrainingResult(
        final_loss=_float_value(payload.get("final_loss"), field_name="final_loss"),
        elapsed_s=_float_value(payload.get("elapsed_s"), field_name="elapsed_s"),
        model_bundle_path=_require_non_empty_str(
            payload.get("model_bundle_path"),
            field_name="model_bundle_path",
        ),
    )


def pipeline_training_status_to_dict(
    status: MorpionPipelineTrainingStatusArtifact,
) -> dict[str, object]:
    """Serialize one training-status artifact into JSON-friendly data."""
    return {
        "evaluator_results": {
            evaluator_name: pipeline_evaluator_training_result_to_dict(result)
            for evaluator_name, result in status.evaluator_results.items()
        },
        "generation": status.generation,
        "metadata": dict(status.metadata),
        "selected_evaluator_name": status.selected_evaluator_name,
        "selection_policy": status.selection_policy,
        "status": status.status,
        "updated_at_utc": status.updated_at_utc,
    }


def pipeline_training_status_from_dict(
    data: object,
) -> MorpionPipelineTrainingStatusArtifact:
    """Deserialize one training-status artifact from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionPipelineTrainingStatusArtifact(
        generation=_require_generation(payload.get("generation"), field_name="generation"),
        status=_training_status(payload.get("status", "not_started")),
        updated_at_utc=_require_non_empty_str(
            payload.get("updated_at_utc"),
            field_name="updated_at_utc",
        ),
        selected_evaluator_name=_optional_str(
            payload.get("selected_evaluator_name"),
            field_name="selected_evaluator_name",
        ),
        selection_policy=_optional_str(
            payload.get("selection_policy"),
            field_name="selection_policy",
        ),
        evaluator_results=_training_result_mapping(payload.get("evaluator_results")),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def pipeline_stage_claim_to_dict(
    claim: MorpionPipelineStageClaim,
) -> dict[str, object]:
    """Serialize one stage-claim record into JSON-friendly data."""
    return {
        "claim_id": claim.claim_id,
        "claimed_at_utc": claim.claimed_at_utc,
        "expires_at_utc": claim.expires_at_utc,
        "generation": claim.generation,
        "metadata": dict(claim.metadata),
        "owner": claim.owner,
        "stage": claim.stage,
    }


def pipeline_stage_claim_from_dict(data: object) -> MorpionPipelineStageClaim:
    """Deserialize one stage-claim record from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionPipelineStageClaim(
        generation=_require_generation(payload.get("generation"), field_name="generation"),
        stage=_stage_name(payload.get("stage")),
        claim_id=_require_str(payload.get("claim_id"), field_name="claim_id"),
        claimed_at_utc=_require_str(
            payload.get("claimed_at_utc"), field_name="claimed_at_utc"
        ),
        expires_at_utc=_require_str(
            payload.get("expires_at_utc"), field_name="expires_at_utc"
        ),
        owner=_optional_str(payload.get("owner"), field_name="owner"),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def reevaluation_patch_row_to_dict(
    row: MorpionReevaluationPatchRow,
) -> dict[str, object]:
    """Serialize one reevaluation-patch row into JSON-friendly data."""
    return {
        "backed_up_value": row.backed_up_value,
        "direct_value": row.direct_value,
        "is_exact": row.is_exact,
        "is_terminal": row.is_terminal,
        "metadata": dict(row.metadata),
        "node_id": row.node_id,
    }


def reevaluation_patch_row_from_dict(data: object) -> MorpionReevaluationPatchRow:
    """Deserialize one reevaluation-patch row from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionReevaluationPatchRow(
        node_id=_require_non_empty_str(payload.get("node_id"), field_name="node_id"),
        direct_value=_float_value(
            payload.get("direct_value"),
            field_name="direct_value",
        ),
        backed_up_value=_optional_float_value(
            payload.get("backed_up_value"),
            field_name="backed_up_value",
        ),
        is_exact=_optional_bool(payload.get("is_exact"), field_name="is_exact"),
        is_terminal=_optional_bool(
            payload.get("is_terminal"),
            field_name="is_terminal",
        ),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def reevaluation_patch_to_dict(
    patch: MorpionReevaluationPatch,
) -> dict[str, object]:
    """Serialize one reevaluation patch into JSON-friendly data."""
    return {
        "created_at_utc": patch.created_at_utc,
        "end_cursor": patch.end_cursor,
        "evaluator_generation": patch.evaluator_generation,
        "evaluator_name": patch.evaluator_name,
        "metadata": dict(patch.metadata),
        "model_bundle_path": patch.model_bundle_path,
        "patch_id": patch.patch_id,
        "rows": [reevaluation_patch_row_to_dict(row) for row in patch.rows],
        "start_cursor": patch.start_cursor,
        "tree_generation": patch.tree_generation,
    }


def reevaluation_patch_from_dict(data: object) -> MorpionReevaluationPatch:
    """Deserialize one reevaluation patch from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionReevaluationPatch(
        patch_id=_require_non_empty_str(payload.get("patch_id"), field_name="patch_id"),
        created_at_utc=_require_non_empty_str(
            payload.get("created_at_utc"),
            field_name="created_at_utc",
        ),
        evaluator_generation=_require_generation(
            payload.get("evaluator_generation"),
            field_name="evaluator_generation",
        ),
        evaluator_name=_require_non_empty_str(
            payload.get("evaluator_name"),
            field_name="evaluator_name",
        ),
        model_bundle_path=_require_non_empty_str(
            payload.get("model_bundle_path"),
            field_name="model_bundle_path",
        ),
        rows=_reevaluation_patch_rows_tuple(payload.get("rows")),
        tree_generation=_optional_generation(
            payload.get("tree_generation"),
            field_name="tree_generation",
        ),
        start_cursor=_optional_str(payload.get("start_cursor"), field_name="start_cursor"),
        end_cursor=_optional_str(payload.get("end_cursor"), field_name="end_cursor"),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def reevaluation_cursor_to_dict(
    cursor: MorpionReevaluationCursor,
) -> dict[str, object]:
    """Serialize one reevaluation cursor into JSON-friendly data."""
    return {
        "completed_full_pass_count": cursor.completed_full_pass_count,
        "evaluator_generation": cursor.evaluator_generation,
        "evaluator_name": cursor.evaluator_name,
        "last_patch_id": cursor.last_patch_id,
        "metadata": dict(cursor.metadata),
        "model_bundle_path": cursor.model_bundle_path,
        "next_node_cursor": cursor.next_node_cursor,
        "tree_generation": cursor.tree_generation,
        "updated_at_utc": cursor.updated_at_utc,
    }


def reevaluation_cursor_from_dict(data: object) -> MorpionReevaluationCursor:
    """Deserialize one reevaluation cursor from JSON-friendly data."""
    payload = _top_level_mapping(data)
    return MorpionReevaluationCursor(
        evaluator_generation=_require_generation(
            payload.get("evaluator_generation"),
            field_name="evaluator_generation",
        ),
        evaluator_name=_require_non_empty_str(
            payload.get("evaluator_name"),
            field_name="evaluator_name",
        ),
        model_bundle_path=_require_non_empty_str(
            payload.get("model_bundle_path"),
            field_name="model_bundle_path",
        ),
        next_node_cursor=_optional_str(
            payload.get("next_node_cursor"),
            field_name="next_node_cursor",
        ),
        updated_at_utc=_require_non_empty_str(
            payload.get("updated_at_utc"),
            field_name="updated_at_utc",
        ),
        tree_generation=_optional_generation(
            payload.get("tree_generation"),
            field_name="tree_generation",
        ),
        completed_full_pass_count=_non_negative_int(
            payload.get("completed_full_pass_count", 0),
            field_name="completed_full_pass_count",
        ),
        last_patch_id=_optional_str(
            payload.get("last_patch_id"),
            field_name="last_patch_id",
        ),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    """Atomically write one JSON payload to disk in a human-readable form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def save_pipeline_manifest(
    manifest: MorpionPipelineGenerationManifest,
    path: Path,
) -> None:
    """Persist one pipeline generation manifest atomically."""
    _atomic_write_json(pipeline_manifest_to_dict(manifest), path)


def load_pipeline_manifest(path: Path) -> MorpionPipelineGenerationManifest:
    """Load one pipeline generation manifest from disk."""
    if not path.is_file():
        raise _missing_manifest_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_manifest_json_error(path) from exc
    return pipeline_manifest_from_dict(payload)


def save_pipeline_active_model(
    active_model: MorpionPipelineActiveModel,
    path: Path,
) -> None:
    """Persist one active-model record atomically."""
    _atomic_write_json(pipeline_active_model_to_dict(active_model), path)


def load_pipeline_active_model(path: Path) -> MorpionPipelineActiveModel:
    """Load one active-model record from disk."""
    if not path.is_file():
        raise _missing_active_model_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_active_model_json_error(path) from exc
    return pipeline_active_model_from_dict(payload)


def load_pipeline_training_status_file(path: Path) -> MorpionPipelineTrainingStatusArtifact:
    """Load one training-status artifact from disk."""
    if not path.is_file():
        raise _missing_manifest_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_manifest_json_error(path) from exc
    return pipeline_training_status_from_dict(payload)


def _missing_stage_claim_error(path: Path) -> MissingMorpionPipelineArtifactError:
    """Return the stable missing-stage-claim error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion pipeline stage-claim artifact does not exist: {path}"
    )


def _invalid_stage_claim_json_error(path: Path) -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-stage-claim-json error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion pipeline stage-claim artifact at {path} is not valid JSON."
    )


def _missing_reevaluation_patch_error(
    path: Path,
) -> MissingMorpionPipelineArtifactError:
    """Return the stable missing-reevaluation-patch error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion reevaluation patch artifact does not exist: {path}"
    )


def _invalid_reevaluation_patch_json_error(
    path: Path,
) -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-reevaluation-patch-json error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion reevaluation patch artifact at {path} is not valid JSON."
    )


def _missing_reevaluation_cursor_error(
    path: Path,
) -> MissingMorpionPipelineArtifactError:
    """Return the stable missing-reevaluation-cursor error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion reevaluation cursor artifact does not exist: {path}"
    )


def _invalid_reevaluation_cursor_json_error(
    path: Path,
) -> InvalidMorpionPipelineArtifactError:
    """Return the stable invalid-reevaluation-cursor-json error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion reevaluation cursor artifact at {path} is not valid JSON."
    )


def save_pipeline_stage_claim(
    claim: MorpionPipelineStageClaim,
    path: Path,
) -> None:
    """Persist one pipeline stage-claim atomically."""
    _atomic_write_json(pipeline_stage_claim_to_dict(claim), path)


def load_pipeline_stage_claim(path: Path) -> MorpionPipelineStageClaim:
    """Load one pipeline stage-claim from disk."""
    if not path.is_file():
        raise _missing_stage_claim_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_stage_claim_json_error(path) from exc
    return pipeline_stage_claim_from_dict(payload)


def delete_pipeline_stage_claim(path: Path) -> None:
    """Delete one persisted pipeline stage-claim if it exists."""
    path.unlink(missing_ok=True)


def save_reevaluation_patch(
    patch: MorpionReevaluationPatch,
    path: Path,
) -> None:
    """Persist one reevaluation patch atomically."""
    _atomic_write_json(reevaluation_patch_to_dict(patch), path)


def load_reevaluation_patch(path: Path) -> MorpionReevaluationPatch:
    """Load one reevaluation patch from disk."""
    if not path.is_file():
        raise _missing_reevaluation_patch_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_reevaluation_patch_json_error(path) from exc
    return reevaluation_patch_from_dict(payload)


def delete_reevaluation_patch(path: Path) -> None:
    """Delete one persisted reevaluation patch if it exists."""
    path.unlink(missing_ok=True)


def save_reevaluation_cursor(
    cursor: MorpionReevaluationCursor,
    path: Path,
) -> None:
    """Persist one reevaluation cursor atomically."""
    _atomic_write_json(reevaluation_cursor_to_dict(cursor), path)


def load_reevaluation_cursor(path: Path) -> MorpionReevaluationCursor:
    """Load one reevaluation cursor from disk."""
    if not path.is_file():
        raise _missing_reevaluation_cursor_error(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _invalid_reevaluation_cursor_json_error(path) from exc
    return reevaluation_cursor_from_dict(payload)


def delete_reevaluation_cursor(path: Path) -> None:
    """Delete one persisted reevaluation cursor if it exists."""
    path.unlink(missing_ok=True)


def save_pipeline_stage_status_file(
    *,
    generation: int,
    status: str,
    updated_at_utc: str,
    metadata: Mapping[str, object] | None,
    path: Path,
) -> None:
    """Persist one lightweight stage-status artifact atomically."""
    _atomic_write_json(
        {
            "generation": _require_generation(generation, field_name="generation"),
            "metadata": _metadata_dict(metadata),
            "status": _require_str(status, field_name="status"),
            "updated_at_utc": _require_str(
                updated_at_utc, field_name="updated_at_utc"
            ),
        },
        path,
    )


def save_pipeline_dataset_status_file(
    *,
    generation: int,
    dataset_status: MorpionPipelineDatasetStatus,
    updated_at_utc: str,
    metadata: Mapping[str, object] | None,
    path: Path,
) -> None:
    """Persist one validated dataset-stage status artifact atomically."""
    save_pipeline_stage_status_file(
        generation=generation,
        status=_dataset_status(dataset_status),
        updated_at_utc=updated_at_utc,
        metadata=metadata,
        path=path,
    )


def save_pipeline_training_status_file(
    *,
    generation: int,
    training_status: MorpionPipelineTrainingStatus,
    updated_at_utc: str,
    metadata: Mapping[str, object] | None,
    selected_evaluator_name: str | None = None,
    selection_policy: str | None = None,
    evaluator_results: Mapping[str, MorpionPipelineEvaluatorTrainingResult] | None = None,
    path: Path,
) -> None:
    """Persist one validated training-stage status artifact atomically."""
    _atomic_write_json(
        pipeline_training_status_to_dict(
            MorpionPipelineTrainingStatusArtifact(
                generation=generation,
                status=_training_status(training_status),
                updated_at_utc=updated_at_utc,
                selected_evaluator_name=selected_evaluator_name,
                selection_policy=selection_policy,
                evaluator_results={}
                if evaluator_results is None
                else dict(evaluator_results),
                metadata=_metadata_dict(metadata),
            )
        ),
        path,
    )


__all__ = [
    "InvalidMorpionPipelineArtifactError",
    "MissingMorpionPipelineArtifactError",
    "MorpionPipelineActiveModel",
    "MorpionPipelineDatasetStatus",
    "MorpionPipelineEvaluatorTrainingResult",
    "MorpionPipelineGenerationManifest",
    "MorpionPipelineStageClaim",
    "MorpionPipelineStageName",
    "MorpionPipelineTrainingStatusArtifact",
    "MorpionPipelineTrainingStatus",
    "MorpionReevaluationCursor",
    "MorpionReevaluationPatch",
    "MorpionReevaluationPatchRow",
    "delete_pipeline_stage_claim",
    "delete_reevaluation_cursor",
    "delete_reevaluation_patch",
    "load_pipeline_active_model",
    "load_pipeline_manifest",
    "load_pipeline_stage_claim",
    "load_pipeline_training_status_file",
    "load_reevaluation_cursor",
    "load_reevaluation_patch",
    "pipeline_active_model_from_dict",
    "pipeline_active_model_to_dict",
    "pipeline_evaluator_training_result_from_dict",
    "pipeline_evaluator_training_result_to_dict",
    "pipeline_manifest_from_dict",
    "pipeline_manifest_to_dict",
    "pipeline_stage_claim_from_dict",
    "pipeline_stage_claim_to_dict",
    "pipeline_training_status_from_dict",
    "pipeline_training_status_to_dict",
    "reevaluation_cursor_from_dict",
    "reevaluation_cursor_to_dict",
    "reevaluation_patch_from_dict",
    "reevaluation_patch_row_from_dict",
    "reevaluation_patch_row_to_dict",
    "reevaluation_patch_to_dict",
    "save_pipeline_active_model",
    "save_pipeline_dataset_status_file",
    "save_pipeline_manifest",
    "save_pipeline_stage_claim",
    "save_pipeline_stage_status_file",
    "save_pipeline_training_status_file",
    "save_reevaluation_cursor",
    "save_reevaluation_patch",
]
