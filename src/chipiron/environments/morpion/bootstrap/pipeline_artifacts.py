"""Durable artifact-contract helpers for the Morpion bootstrap pipeline."""

from __future__ import annotations

import json
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
    path: Path,
) -> None:
    """Persist one validated training-stage status artifact atomically."""
    save_pipeline_stage_status_file(
        generation=generation,
        status=_training_status(training_status),
        updated_at_utc=updated_at_utc,
        metadata=metadata,
        path=path,
    )


__all__ = [
    "InvalidMorpionPipelineArtifactError",
    "MissingMorpionPipelineArtifactError",
    "MorpionPipelineActiveModel",
    "MorpionPipelineDatasetStatus",
    "MorpionPipelineGenerationManifest",
    "MorpionPipelineStageClaim",
    "MorpionPipelineStageName",
    "MorpionPipelineTrainingStatus",
    "delete_pipeline_stage_claim",
    "load_pipeline_active_model",
    "load_pipeline_manifest",
    "load_pipeline_stage_claim",
    "pipeline_active_model_from_dict",
    "pipeline_active_model_to_dict",
    "pipeline_manifest_from_dict",
    "pipeline_manifest_to_dict",
    "pipeline_stage_claim_from_dict",
    "pipeline_stage_claim_to_dict",
    "save_pipeline_active_model",
    "save_pipeline_dataset_status_file",
    "save_pipeline_manifest",
    "save_pipeline_stage_claim",
    "save_pipeline_stage_status_file",
    "save_pipeline_training_status_file",
]
