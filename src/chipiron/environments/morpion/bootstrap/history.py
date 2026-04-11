"""Persistent history and latest-status helpers for Morpion bootstrap runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


def _empty_evaluators() -> dict[str, MorpionEvaluatorMetrics]:
    """Return a typed empty evaluator mapping."""
    return {}


def _empty_model_bundle_paths() -> dict[str, str]:
    """Return a typed empty model-bundle-path mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorMetrics:
    """Metrics for one evaluator trained during a bootstrap cycle."""

    final_loss: float | None
    num_epochs: int | None
    num_samples: int | None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapTreeStatus:
    """Tree-related monitoring fields for one bootstrap cycle."""

    size: int
    size_at_last_save: int | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapDatasetStatus:
    """Dataset-related monitoring fields for one bootstrap cycle."""

    rows_count: int | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapTrainingStatus:
    """Training-related monitoring fields for one bootstrap cycle."""

    triggered: bool


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRecordStatus:
    """Record-related monitoring fields for one bootstrap cycle."""

    current: float | int | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapArtifacts:
    """Artifact paths produced by one bootstrap cycle."""

    tree_snapshot_path: str | None
    rows_path: str | None
    model_bundle_paths: dict[str, str] = field(default_factory=_empty_model_bundle_paths)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapEvent:
    """One persisted bootstrap-cycle monitoring event."""

    event_id: str = field(default_factory=lambda: uuid4().hex)
    cycle_index: int = 0
    generation: int = 0
    timestamp_utc: str = ""
    tree: MorpionBootstrapTreeStatus = field(
        default_factory=lambda: MorpionBootstrapTreeStatus(size=0, size_at_last_save=None)
    )
    dataset: MorpionBootstrapDatasetStatus = field(
        default_factory=lambda: MorpionBootstrapDatasetStatus(rows_count=None)
    )
    training: MorpionBootstrapTrainingStatus = field(
        default_factory=lambda: MorpionBootstrapTrainingStatus(triggered=False)
    )
    record: MorpionBootstrapRecordStatus = field(
        default_factory=lambda: MorpionBootstrapRecordStatus(current=None)
    )
    artifacts: MorpionBootstrapArtifacts = field(
        default_factory=lambda: MorpionBootstrapArtifacts(
            tree_snapshot_path=None,
            rows_path=None,
        )
    )
    evaluators: dict[str, MorpionEvaluatorMetrics] = field(default_factory=_empty_evaluators)
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapLatestStatus:
    """Fast-loading latest known bootstrap status for the future GUI."""

    work_dir: str
    latest_generation: int | None
    latest_cycle_index: int | None
    latest_event: MorpionBootstrapEvent | None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapHistoryPaths:
    """Canonical paths for persisted bootstrap history artifacts."""

    work_dir: Path
    history_jsonl_path: Path
    latest_status_path: Path


class MalformedMorpionBootstrapHistoryError(TypeError):
    """Raised when bootstrap history or latest-status payloads are malformed."""

    @classmethod
    def invalid_event_mapping(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid-event-mapping error."""
        return cls("Morpion bootstrap event payload must be a mapping with string keys.")

    @classmethod
    def invalid_section_mapping(
        cls,
        section_name: str,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid nested-section mapping error."""
        return cls(
            f"Morpion bootstrap history section `{section_name}` must be a mapping "
            "with string keys."
        )

    @classmethod
    def invalid_evaluator_mapping(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid-evaluator-mapping error."""
        return cls(
            "Morpion bootstrap evaluator metrics payload must be a mapping with string keys."
        )

    @classmethod
    def invalid_latest_status_mapping(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid-latest-status-mapping error."""
        return cls(
            "Morpion bootstrap latest-status payload must be a mapping with string keys."
        )

    @classmethod
    def invalid_optional_str_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid optional-string field error."""
        return cls(
            f"Morpion bootstrap history field `{field_name}` must be a string or null."
        )

    @classmethod
    def invalid_required_str_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid required-string field error."""
        return cls(
            f"Morpion bootstrap history field `{field_name}` must be a string."
        )

    @classmethod
    def invalid_bool_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid bool field error."""
        return cls(f"Morpion bootstrap history field `{field_name}` must be a bool.")

    @classmethod
    def invalid_metadata(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid metadata error."""
        return cls("Morpion bootstrap history field `metadata` must be a mapping.")

    @classmethod
    def invalid_evaluators_field(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid evaluators field error."""
        return cls("Morpion bootstrap history field `evaluators` must be a mapping.")

    @classmethod
    def invalid_model_bundle_paths_field(
        cls,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid model-bundle-paths field error."""
        return cls(
            "Morpion bootstrap history field `artifacts.model_bundle_paths` must be a "
            "mapping of strings to strings."
        )

    @classmethod
    def malformed_history_line(
        cls,
        line_number: int,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the malformed history line error."""
        return cls(f"Morpion bootstrap history line {line_number} is malformed.")

    @classmethod
    def invalid_integer_like_value(
        cls,
        field_name: str,
        value: object,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid integer-like payload error."""
        return cls(
            f"Morpion bootstrap history field `{field_name}` must be integer-like, "
            f"got {type(value).__name__}."
        )

    @classmethod
    def invalid_float_like_value(
        cls,
        field_name: str,
        value: object,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid float-like payload error."""
        return cls(
            f"Morpion bootstrap history field `{field_name}` must be float-like, "
            f"got {type(value).__name__}."
        )

    @classmethod
    def invalid_optional_number(
        cls,
        field_name: str,
        value: object,
    ) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid optional-number payload error."""
        return cls(
            f"Morpion bootstrap history field `{field_name}` must be numeric or null, "
            f"got {type(value).__name__}."
        )


def bootstrap_event_to_dict(event: MorpionBootstrapEvent) -> dict[str, object]:
    """Serialize one bootstrap event to JSON-friendly data."""
    return {
        "event_id": event.event_id,
        "cycle_index": event.cycle_index,
        "generation": event.generation,
        "timestamp_utc": event.timestamp_utc,
        "tree": {
            "size": event.tree.size,
            "size_at_last_save": event.tree.size_at_last_save,
        },
        "dataset": {"rows_count": event.dataset.rows_count},
        "training": {"triggered": event.training.triggered},
        "record": {"current": event.record.current},
        "artifacts": {
            "tree_snapshot_path": event.artifacts.tree_snapshot_path,
            "rows_path": event.artifacts.rows_path,
            "model_bundle_paths": dict(event.artifacts.model_bundle_paths),
        },
        "evaluators": {
            name: evaluator_metrics_to_dict(metrics)
            for name, metrics in event.evaluators.items()
        },
        "metadata": dict(event.metadata),
    }


def bootstrap_event_from_dict(data: dict[str, object]) -> MorpionBootstrapEvent:
    """Deserialize one bootstrap event from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapHistoryError.invalid_event_mapping()

    tree_data = _require_section_mapping(data.get("tree"), section_name="tree")
    dataset_data = _require_section_mapping(data.get("dataset"), section_name="dataset")
    training_data = _require_section_mapping(
        data.get("training"),
        section_name="training",
    )
    record_data = _require_section_mapping(data.get("record"), section_name="record")
    artifacts_data = _require_section_mapping(
        data.get("artifacts"),
        section_name="artifacts",
    )
    evaluators_data = data.get("evaluators", {})
    if not _is_str_key_mapping(evaluators_data):
        raise MalformedMorpionBootstrapHistoryError.invalid_evaluators_field()

    return MorpionBootstrapEvent(
        event_id=_required_str(data.get("event_id"), field_name="event_id"),
        cycle_index=_coerce_int(data.get("cycle_index"), field_name="cycle_index"),
        generation=_coerce_int(data.get("generation"), field_name="generation"),
        timestamp_utc=_required_str(
            data.get("timestamp_utc"),
            field_name="timestamp_utc",
        ),
        tree=MorpionBootstrapTreeStatus(
            size=_coerce_int(tree_data.get("size"), field_name="tree.size"),
            size_at_last_save=_optional_int(
                tree_data.get("size_at_last_save"),
                field_name="tree.size_at_last_save",
            ),
        ),
        dataset=MorpionBootstrapDatasetStatus(
            rows_count=_optional_int(
                dataset_data.get("rows_count"),
                field_name="dataset.rows_count",
            )
        ),
        training=MorpionBootstrapTrainingStatus(
            triggered=_required_bool(
                training_data.get("triggered"),
                field_name="training.triggered",
            )
        ),
        record=MorpionBootstrapRecordStatus(
            current=_optional_number(
                record_data.get("current"),
                field_name="record.current",
            )
        ),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=_optional_str(
                artifacts_data.get("tree_snapshot_path"),
                field_name="artifacts.tree_snapshot_path",
            ),
            rows_path=_optional_str(
                artifacts_data.get("rows_path"),
                field_name="artifacts.rows_path",
            ),
            model_bundle_paths=_string_mapping(
                artifacts_data.get("model_bundle_paths"),
                field_name="artifacts.model_bundle_paths",
            ),
        ),
        evaluators={
            name: evaluator_metrics_from_dict(_require_evaluator_mapping(item))
            for name, item in cast("Mapping[str, object]", evaluators_data).items()
        },
        metadata=_metadata_dict(data.get("metadata")),
    )


def evaluator_metrics_to_dict(metrics: MorpionEvaluatorMetrics) -> dict[str, object]:
    """Serialize one evaluator metrics record to JSON-friendly data."""
    return {
        "final_loss": metrics.final_loss,
        "num_epochs": metrics.num_epochs,
        "num_samples": metrics.num_samples,
        "metadata": dict(metrics.metadata),
    }


def evaluator_metrics_from_dict(
    data: dict[str, object],
) -> MorpionEvaluatorMetrics:
    """Deserialize one evaluator metrics record from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapHistoryError.invalid_evaluator_mapping()

    return MorpionEvaluatorMetrics(
        final_loss=_optional_float(
            data.get("final_loss"),
            field_name="final_loss",
        ),
        num_epochs=_optional_int(
            data.get("num_epochs"),
            field_name="num_epochs",
        ),
        num_samples=_optional_int(
            data.get("num_samples"),
            field_name="num_samples",
        ),
        metadata=_metadata_dict(data.get("metadata")),
    )


def latest_status_to_dict(
    status: MorpionBootstrapLatestStatus,
) -> dict[str, object]:
    """Serialize one latest-status payload to JSON-friendly data."""
    return {
        "work_dir": status.work_dir,
        "latest_generation": status.latest_generation,
        "latest_cycle_index": status.latest_cycle_index,
        "latest_event": None
        if status.latest_event is None
        else bootstrap_event_to_dict(status.latest_event),
        "metadata": dict(status.metadata),
    }


def latest_status_from_dict(
    data: dict[str, object],
) -> MorpionBootstrapLatestStatus:
    """Deserialize one latest-status payload from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapHistoryError.invalid_latest_status_mapping()

    latest_event_data = data.get("latest_event")
    if latest_event_data is None:
        latest_event = None
    else:
        latest_event = bootstrap_event_from_dict(_require_event_mapping(latest_event_data))

    latest_generation = _optional_int(
        data.get("latest_generation"),
        field_name="latest_generation",
    )
    latest_cycle_index = _optional_int(
        data.get("latest_cycle_index"),
        field_name="latest_cycle_index",
    )

    if latest_event is not None:
        if latest_generation is None:
            latest_generation = latest_event.generation
        if latest_cycle_index is None:
            latest_cycle_index = latest_event.cycle_index

    return MorpionBootstrapLatestStatus(
        work_dir=_required_str(data.get("work_dir"), field_name="work_dir"),
        latest_generation=latest_generation,
        latest_cycle_index=latest_cycle_index,
        latest_event=latest_event,
        metadata=_metadata_dict(data.get("metadata")),
    )


class MorpionBootstrapHistoryRecorder:
    """Append-only history recorder plus latest-status writer."""

    def __init__(self, paths: MorpionBootstrapHistoryPaths) -> None:
        """Store canonical paths for bootstrap history artifacts."""
        self.paths = paths

    def append_event(self, event: MorpionBootstrapEvent) -> None:
        """Append one JSONL bootstrap event."""
        self.paths.history_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.history_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap_event_to_dict(event), sort_keys=True))
            handle.write("\n")

    def write_latest_status(self, status: MorpionBootstrapLatestStatus) -> None:
        """Overwrite the latest bootstrap status snapshot atomically."""
        _write_json_atomic(
            self.paths.latest_status_path,
            latest_status_to_dict(status),
        )

    def record(self, event: MorpionBootstrapEvent) -> None:
        """Append one event and refresh the latest-status snapshot."""
        self.append_event(event)
        self.write_latest_status(
            MorpionBootstrapLatestStatus(
                work_dir=str(self.paths.work_dir),
                latest_generation=event.generation,
                latest_cycle_index=event.cycle_index,
                latest_event=event,
            )
        )


def load_bootstrap_history(
    path: str | Path,
) -> tuple[MorpionBootstrapEvent, ...]:
    """Load every non-empty bootstrap history event from a JSONL file."""
    events: list[MorpionBootstrapEvent] = []
    history_path = Path(path)
    if not history_path.exists():
        return ()

    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
            event = bootstrap_event_from_dict(_require_event_mapping(loaded))
        except (json.JSONDecodeError, MalformedMorpionBootstrapHistoryError) as exc:
            raise MalformedMorpionBootstrapHistoryError.malformed_history_line(
                line_number
            ) from exc
        events.append(event)
    return tuple(events)


def load_latest_bootstrap_status(
    path: str | Path,
) -> MorpionBootstrapLatestStatus:
    """Load the latest bootstrap status snapshot from JSON."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MalformedMorpionBootstrapHistoryError.invalid_latest_status_mapping() from exc
    return latest_status_from_dict(_require_latest_status_mapping(loaded))


def rebuild_latest_bootstrap_status(
    paths: MorpionBootstrapHistoryPaths,
) -> MorpionBootstrapLatestStatus:
    """Rebuild ``latest_status.json`` from the append-only history."""
    history = load_bootstrap_history(paths.history_jsonl_path)
    latest_event = history[-1] if history else None
    status = MorpionBootstrapLatestStatus(
        work_dir=str(paths.work_dir),
        latest_generation=None if latest_event is None else latest_event.generation,
        latest_cycle_index=None if latest_event is None else latest_event.cycle_index,
        latest_event=latest_event,
    )
    MorpionBootstrapHistoryRecorder(paths).write_latest_status(status)
    return status


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _require_event_mapping(value: object) -> dict[str, object]:
    """Return one event mapping or raise."""
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_event_mapping()
    return dict(cast("Mapping[str, object]", value))


def _require_evaluator_mapping(value: object) -> dict[str, object]:
    """Return one evaluator mapping or raise."""
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_evaluator_mapping()
    return dict(cast("Mapping[str, object]", value))


def _require_latest_status_mapping(value: object) -> dict[str, object]:
    """Return one latest-status mapping or raise."""
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_latest_status_mapping()
    return dict(cast("Mapping[str, object]", value))


def _require_section_mapping(
    value: object,
    *,
    section_name: str,
) -> dict[str, object]:
    """Return one nested-section mapping or raise."""
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_section_mapping(section_name)
    return dict(cast("Mapping[str, object]", value))


def _is_str_key_mapping(value: object) -> bool:
    """Return whether ``value`` is a mapping with string keys."""
    if not isinstance(value, Mapping):
        return False
    mapping = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in mapping)


def _required_str(value: object, *, field_name: str) -> str:
    """Return one required string field or raise."""
    if isinstance(value, str):
        return value
    raise MalformedMorpionBootstrapHistoryError.invalid_required_str_field(field_name)


def _optional_str(value: object, *, field_name: str) -> str | None:
    """Return one optional string field or raise."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise MalformedMorpionBootstrapHistoryError.invalid_optional_str_field(field_name)


def _required_bool(value: object, *, field_name: str) -> bool:
    """Return one required bool field or raise."""
    if isinstance(value, bool):
        return value
    raise MalformedMorpionBootstrapHistoryError.invalid_bool_field(field_name)


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return one metadata dictionary or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_metadata()
    return dict(cast("Mapping[str, Any]", value))


def _string_mapping(value: object, *, field_name: str) -> dict[str, str]:
    """Return one string-to-string mapping or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_model_bundle_paths_field()

    raw_mapping = cast("Mapping[str, object]", value)
    if not all(isinstance(item_value, str) for item_value in raw_mapping.values()):
        raise MalformedMorpionBootstrapHistoryError.invalid_model_bundle_paths_field()

    _ = field_name
    return {key: cast("str", item_value) for key, item_value in raw_mapping.items()}


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return one integer-like payload value or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapHistoryError.invalid_integer_like_value(
            field_name,
            value,
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise MalformedMorpionBootstrapHistoryError.invalid_integer_like_value(
            field_name,
            value,
        )
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise MalformedMorpionBootstrapHistoryError.invalid_integer_like_value(
                field_name,
                value,
            ) from exc
    raise MalformedMorpionBootstrapHistoryError.invalid_integer_like_value(
        field_name,
        value,
    )


def _optional_int(value: object, *, field_name: str) -> int | None:
    """Return one optional integer-like payload value or raise."""
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _coerce_float(value: object, *, field_name: str) -> float:
    """Return one float-like payload value or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapHistoryError.invalid_float_like_value(
            field_name,
            value,
        )
    try:
        if isinstance(value, int | float | str):
            return float(value)
    except ValueError as exc:
        raise MalformedMorpionBootstrapHistoryError.invalid_float_like_value(
            field_name,
            value,
        ) from exc
    raise MalformedMorpionBootstrapHistoryError.invalid_float_like_value(
        field_name,
        value,
    )


def _optional_float(value: object, *, field_name: str) -> float | None:
    """Return one optional float-like payload value or raise."""
    if value is None:
        return None
    return _coerce_float(value, field_name=field_name)


def _optional_number(value: object, *, field_name: str) -> int | float | None:
    """Return one optional numeric payload value or raise."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapHistoryError.invalid_optional_number(
            field_name,
            value,
        )
    if isinstance(value, int | float):
        return value
    raise MalformedMorpionBootstrapHistoryError.invalid_optional_number(
        field_name,
        value,
    )


__all__ = [
    "MalformedMorpionBootstrapHistoryError",
    "MorpionBootstrapArtifacts",
    "MorpionBootstrapDatasetStatus",
    "MorpionBootstrapEvent",
    "MorpionBootstrapHistoryPaths",
    "MorpionBootstrapHistoryRecorder",
    "MorpionBootstrapLatestStatus",
    "MorpionBootstrapRecordStatus",
    "MorpionBootstrapTrainingStatus",
    "MorpionBootstrapTreeStatus",
    "MorpionEvaluatorMetrics",
    "bootstrap_event_from_dict",
    "bootstrap_event_to_dict",
    "evaluator_metrics_from_dict",
    "evaluator_metrics_to_dict",
    "latest_status_from_dict",
    "latest_status_to_dict",
    "load_bootstrap_history",
    "load_latest_bootstrap_status",
    "rebuild_latest_bootstrap_status",
]
