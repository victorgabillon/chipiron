"""Persistent history and latest-status helpers for Morpion bootstrap runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorMetrics:
    """Metrics for one evaluator trained during a bootstrap cycle."""

    name: str
    model_bundle_path: str | None
    final_loss: float | None
    num_epochs: int | None
    num_samples: int | None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapEvent:
    """One persisted bootstrap-cycle monitoring event."""

    timestamp_unix_s: float
    generation: int
    tree_size: int
    tree_size_at_last_save: int | None
    tree_snapshot_path: str | None
    rows_path: str | None
    rows_count: int | None
    current_record: float | int | None
    evaluators: tuple[MorpionEvaluatorMetrics, ...] = ()
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapLatestStatus:
    """Fast-loading latest known bootstrap status for the future GUI."""

    latest_event: MorpionBootstrapEvent | None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapHistoryPaths:
    """Canonical paths for persisted bootstrap history artifacts."""

    history_jsonl_path: Path
    latest_status_path: Path


class MalformedMorpionBootstrapHistoryError(TypeError):
    """Raised when bootstrap history or latest-status payloads are malformed."""

    @classmethod
    def invalid_event_mapping(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid-event-mapping error."""
        return cls("Morpion bootstrap event payload must be a mapping with string keys.")

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
    def invalid_metadata(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid metadata error."""
        return cls("Morpion bootstrap history field `metadata` must be a mapping.")

    @classmethod
    def invalid_evaluators_field(cls) -> MalformedMorpionBootstrapHistoryError:
        """Return the invalid evaluators field error."""
        return cls("Morpion bootstrap history field `evaluators` must be a list.")

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
        "timestamp_unix_s": event.timestamp_unix_s,
        "generation": event.generation,
        "tree_size": event.tree_size,
        "tree_size_at_last_save": event.tree_size_at_last_save,
        "tree_snapshot_path": event.tree_snapshot_path,
        "rows_path": event.rows_path,
        "rows_count": event.rows_count,
        "current_record": event.current_record,
        "evaluators": [evaluator_metrics_to_dict(item) for item in event.evaluators],
        "metadata": dict(event.metadata),
    }


def bootstrap_event_from_dict(data: dict[str, object]) -> MorpionBootstrapEvent:
    """Deserialize one bootstrap event from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapHistoryError.invalid_event_mapping()
    evaluators_data = data.get("evaluators", [])
    if not isinstance(evaluators_data, list):
        raise MalformedMorpionBootstrapHistoryError.invalid_evaluators_field()

    return MorpionBootstrapEvent(
        timestamp_unix_s=_coerce_float(
            data.get("timestamp_unix_s"),
            field_name="timestamp_unix_s",
        ),
        generation=_coerce_int(data.get("generation"), field_name="generation"),
        tree_size=_coerce_int(data.get("tree_size"), field_name="tree_size"),
        tree_size_at_last_save=_optional_int(
            data.get("tree_size_at_last_save"),
            field_name="tree_size_at_last_save",
        ),
        tree_snapshot_path=_optional_str(
            data.get("tree_snapshot_path"),
            field_name="tree_snapshot_path",
        ),
        rows_path=_optional_str(
            data.get("rows_path"),
            field_name="rows_path",
        ),
        rows_count=_optional_int(
            data.get("rows_count"),
            field_name="rows_count",
        ),
        current_record=_optional_number(
            data.get("current_record"),
            field_name="current_record",
        ),
        evaluators=tuple(
            evaluator_metrics_from_dict(_require_evaluator_mapping(item))
            for item in evaluators_data
        ),
        metadata=_metadata_dict(data.get("metadata")),
    )


def evaluator_metrics_to_dict(metrics: MorpionEvaluatorMetrics) -> dict[str, object]:
    """Serialize one evaluator metrics record to JSON-friendly data."""
    return {
        "name": metrics.name,
        "model_bundle_path": metrics.model_bundle_path,
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
        name=_required_str(data.get("name"), field_name="name"),
        model_bundle_path=_optional_str(
            data.get("model_bundle_path"),
            field_name="model_bundle_path",
        ),
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

    return MorpionBootstrapLatestStatus(
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
        """Overwrite the latest bootstrap status snapshot."""
        self.paths.latest_status_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.latest_status_path.write_text(
            json.dumps(latest_status_to_dict(status), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(self, event: MorpionBootstrapEvent) -> None:
        """Append one event and refresh the latest-status snapshot."""
        self.append_event(event)
        self.write_latest_status(MorpionBootstrapLatestStatus(latest_event=event))


def load_bootstrap_history(
    path: str | Path,
) -> tuple[MorpionBootstrapEvent, ...]:
    """Load every non-empty bootstrap history event from a JSONL file."""
    events: list[MorpionBootstrapEvent] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
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


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return one metadata dictionary or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapHistoryError.invalid_metadata()
    return dict(cast("Mapping[str, Any]", value))


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return one integer-like payload value or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapHistoryError.invalid_integer_like_value(
            field_name,
            value,
        )
    try:
        if isinstance(value, int | str):
            return int(value)
        if isinstance(value, float):
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
    "MorpionBootstrapEvent",
    "MorpionBootstrapHistoryPaths",
    "MorpionBootstrapHistoryRecorder",
    "MorpionBootstrapLatestStatus",
    "MorpionEvaluatorMetrics",
    "bootstrap_event_from_dict",
    "bootstrap_event_to_dict",
    "evaluator_metrics_from_dict",
    "evaluator_metrics_to_dict",
    "latest_status_from_dict",
    "latest_status_to_dict",
    "load_bootstrap_history",
    "load_latest_bootstrap_status",
]
