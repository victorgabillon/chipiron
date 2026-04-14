"""Persistence helpers for Morpion bootstrap run state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .record_status import (
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
)


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRunState:
    """Persisted state for a restartable Morpion bootstrap run."""

    generation: int
    cycle_index: int
    latest_tree_snapshot_path: str | None
    latest_rows_path: str | None
    latest_model_bundle_paths: dict[str, str] | None
    active_evaluator_name: str | None
    tree_size_at_last_save: int
    last_save_unix_s: float | None
    latest_runtime_checkpoint_path: str | None = None
    latest_record_status: MorpionBootstrapRecordStatus | None = None
    latest_frontier_status: MorpionBootstrapFrontierStatus | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


class MalformedMorpionBootstrapRunStateError(TypeError):
    """Raised when persisted Morpion bootstrap state is structurally malformed."""

    @classmethod
    def invalid_top_level_mapping(
        cls,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid top-level payload error."""
        return cls("Morpion bootstrap run state must be a mapping with string keys.")

    @classmethod
    def invalid_optional_str_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid optional-string field error."""
        return cls(
            f"Morpion bootstrap run state field `{field_name}` must be a string or null."
        )

    @classmethod
    def invalid_optional_str_mapping_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid optional string-mapping field error."""
        return cls(
            f"Morpion bootstrap run state field `{field_name}` must be a mapping "
            "of strings to strings or null."
        )

    @classmethod
    def invalid_metadata(
        cls,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid metadata field error."""
        return cls("Morpion bootstrap run state field `metadata` must be a mapping.")

    @classmethod
    def invalid_integer_like_value(
        cls,
        field_name: str,
        value: object,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid integer-like payload error."""
        return cls(
            f"Morpion bootstrap run state field `{field_name}` must be integer-like, "
            f"got {type(value).__name__}."
        )

    @classmethod
    def invalid_float_like_value(
        cls,
        field_name: str,
        value: object,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid float-like payload error."""
        return cls(
            f"Morpion bootstrap run state field `{field_name}` must be float-like or "
            f"null, got {type(value).__name__}."
        )

    @classmethod
    def invalid_bool_field(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid bool field error."""
        return cls(f"Morpion bootstrap run state field `{field_name}` must be a bool.")

    @classmethod
    def invalid_record_status_field(
        cls,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid record-status field error."""
        return cls(
            "Morpion bootstrap run state field `latest_record_status` must be a "
            "mapping with string keys or null."
        )

    @classmethod
    def invalid_frontier_status_field(
        cls,
    ) -> MalformedMorpionBootstrapRunStateError:
        """Return the invalid frontier-status field error."""
        return cls(
            "Morpion bootstrap run state field `latest_frontier_status` must be a "
            "mapping with string keys or null."
        )


def initialize_bootstrap_run_state() -> MorpionBootstrapRunState:
    """Return the initial empty run state for a fresh bootstrap run."""
    return MorpionBootstrapRunState(
        generation=0,
        cycle_index=-1,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
        latest_runtime_checkpoint_path=None,
        latest_record_status=None,
        latest_frontier_status=None,
    )


def save_bootstrap_run_state(
    state: MorpionBootstrapRunState,
    path: str | Path,
) -> None:
    """Persist one Morpion bootstrap run state as UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_run_state_to_dict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_bootstrap_run_state(
    path: str | Path,
) -> MorpionBootstrapRunState:
    """Load one Morpion bootstrap run state from UTF-8 JSON."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return _run_state_from_dict(loaded)


def _run_state_to_dict(state: MorpionBootstrapRunState) -> dict[str, object]:
    """Serialize one run state to JSON-friendly data."""
    return {
        "generation": state.generation,
        "cycle_index": state.cycle_index,
        "latest_tree_snapshot_path": state.latest_tree_snapshot_path,
        "latest_rows_path": state.latest_rows_path,
        "latest_model_bundle_paths": None
        if state.latest_model_bundle_paths is None
        else dict(state.latest_model_bundle_paths),
        "active_evaluator_name": state.active_evaluator_name,
        "tree_size_at_last_save": state.tree_size_at_last_save,
        "last_save_unix_s": state.last_save_unix_s,
        "latest_runtime_checkpoint_path": state.latest_runtime_checkpoint_path,
        "latest_record_status": None
        if state.latest_record_status is None
        else _record_status_to_dict(state.latest_record_status),
        "latest_frontier_status": None
        if state.latest_frontier_status is None
        else _frontier_status_to_dict(state.latest_frontier_status),
        "metadata": dict(state.metadata),
    }


def _run_state_from_dict(data: object) -> MorpionBootstrapRunState:
    """Deserialize one run state from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapRunStateError.invalid_top_level_mapping()

    payload = cast("Mapping[str, object]", data)
    latest_model_bundle_paths = _optional_str_mapping(
        payload.get("latest_model_bundle_paths"),
        field_name="latest_model_bundle_paths",
    )
    legacy_latest_model_bundle_path = _optional_str(
        payload.get("latest_model_bundle_path"),
        field_name="latest_model_bundle_path",
    )
    if (
        latest_model_bundle_paths is None
        and legacy_latest_model_bundle_path is not None
    ):
        latest_model_bundle_paths = {"default": legacy_latest_model_bundle_path}
    active_evaluator_name = _optional_str(
        payload.get("active_evaluator_name"),
        field_name="active_evaluator_name",
    )
    if (
        active_evaluator_name is None
        and latest_model_bundle_paths is not None
        and len(latest_model_bundle_paths) == 1
    ):
        active_evaluator_name = next(iter(latest_model_bundle_paths))
    latest_record_status_data = payload.get("latest_record_status")
    if latest_record_status_data is None:
        latest_record_status = None
    else:
        if not _is_str_key_mapping(latest_record_status_data):
            raise MalformedMorpionBootstrapRunStateError.invalid_record_status_field()
        latest_record_status = _record_status_from_dict(
            cast("Mapping[str, object]", latest_record_status_data)
        )
    latest_frontier_status_data = payload.get("latest_frontier_status")
    if latest_frontier_status_data is None:
        latest_frontier_status = None
    else:
        if not _is_str_key_mapping(latest_frontier_status_data):
            raise MalformedMorpionBootstrapRunStateError.invalid_frontier_status_field()
        latest_frontier_status = _frontier_status_from_dict(
            cast("Mapping[str, object]", latest_frontier_status_data)
        )

    return MorpionBootstrapRunState(
        generation=_coerce_int(payload.get("generation", 0), field_name="generation"),
        cycle_index=_coerce_int(
            payload.get("cycle_index", -1), field_name="cycle_index"
        ),
        latest_tree_snapshot_path=_optional_str(
            payload.get("latest_tree_snapshot_path"),
            field_name="latest_tree_snapshot_path",
        ),
        latest_rows_path=_optional_str(
            payload.get("latest_rows_path"),
            field_name="latest_rows_path",
        ),
        latest_model_bundle_paths=latest_model_bundle_paths,
        active_evaluator_name=active_evaluator_name,
        tree_size_at_last_save=_coerce_int(
            payload.get("tree_size_at_last_save", 0),
            field_name="tree_size_at_last_save",
        ),
        last_save_unix_s=_optional_float(
            payload.get("last_save_unix_s"),
            field_name="last_save_unix_s",
        ),
        latest_runtime_checkpoint_path=_optional_str(
            payload.get("latest_runtime_checkpoint_path"),
            field_name="latest_runtime_checkpoint_path",
        ),
        latest_record_status=latest_record_status,
        latest_frontier_status=latest_frontier_status,
        metadata=_metadata_dict(payload.get("metadata")),
    )


def _is_str_key_mapping(value: object) -> bool:
    """Return whether ``value`` is a mapping with string keys."""
    if not isinstance(value, Mapping):
        return False
    mapping = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in mapping)


def _optional_str(value: object, *, field_name: str) -> str | None:
    """Return one optional string field or raise."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise MalformedMorpionBootstrapRunStateError.invalid_optional_str_field(field_name)


def _optional_str_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, str] | None:
    """Return one optional string-to-string mapping field or raise."""
    if value is None:
        return None
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapRunStateError.invalid_optional_str_mapping_field(
            field_name
        )
    mapping = cast("Mapping[str, object]", value)
    if not all(isinstance(item_value, str) for item_value in mapping.values()):
        raise MalformedMorpionBootstrapRunStateError.invalid_optional_str_mapping_field(
            field_name
        )
    return {key: cast("str", item_value) for key, item_value in mapping.items()}


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return one metadata dictionary or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapRunStateError.invalid_metadata()
    return dict(cast("Mapping[str, Any]", value))


def _record_status_to_dict(
    status: MorpionBootstrapRecordStatus,
) -> dict[str, object]:
    """Serialize one Morpion record status into JSON-friendly data."""
    return {
        "variant": status.variant,
        "initial_pattern": status.initial_pattern,
        "initial_point_count": status.initial_point_count,
        "current_best_moves_since_start": status.current_best_moves_since_start,
        "current_best_total_points": status.current_best_total_points,
        "current_best_is_exact": status.current_best_is_exact,
        "current_best_is_terminal": status.current_best_is_terminal,
        "current_best_source": status.current_best_source,
    }


def _record_status_from_dict(
    data: Mapping[str, object],
) -> MorpionBootstrapRecordStatus:
    """Deserialize one Morpion record status from JSON-friendly data."""
    return MorpionBootstrapRecordStatus(
        variant=_optional_str(
            data.get("variant"),
            field_name="latest_record_status.variant",
        ),
        initial_pattern=_optional_str(
            data.get("initial_pattern"),
            field_name="latest_record_status.initial_pattern",
        ),
        initial_point_count=_optional_int(
            data.get("initial_point_count"),
            field_name="latest_record_status.initial_point_count",
        ),
        current_best_moves_since_start=_optional_int(
            data.get("current_best_moves_since_start"),
            field_name="latest_record_status.current_best_moves_since_start",
        ),
        current_best_total_points=_optional_int(
            data.get("current_best_total_points"),
            field_name="latest_record_status.current_best_total_points",
        ),
        current_best_is_exact=_optional_bool(
            data.get("current_best_is_exact"),
            field_name="latest_record_status.current_best_is_exact",
        ),
        current_best_is_terminal=_optional_bool(
            data.get("current_best_is_terminal"),
            field_name="latest_record_status.current_best_is_terminal",
        ),
        current_best_source=_optional_str(
            data.get("current_best_source"),
            field_name="latest_record_status.current_best_source",
        ),
    )


def _frontier_status_to_dict(
    status: MorpionBootstrapFrontierStatus,
) -> dict[str, object]:
    """Serialize one Morpion frontier status into JSON-friendly data."""
    return {
        "variant": status.variant,
        "initial_pattern": status.initial_pattern,
        "initial_point_count": status.initial_point_count,
        "current_best_moves_since_start": status.current_best_moves_since_start,
        "current_best_total_points": status.current_best_total_points,
        "current_best_is_exact": status.current_best_is_exact,
        "current_best_is_terminal": status.current_best_is_terminal,
        "current_best_source": status.current_best_source,
    }


def _frontier_status_from_dict(
    data: Mapping[str, object],
) -> MorpionBootstrapFrontierStatus:
    """Deserialize one Morpion frontier status from JSON-friendly data."""
    return MorpionBootstrapFrontierStatus(
        variant=_optional_str(
            data.get("variant"),
            field_name="latest_frontier_status.variant",
        ),
        initial_pattern=_optional_str(
            data.get("initial_pattern"),
            field_name="latest_frontier_status.initial_pattern",
        ),
        initial_point_count=_optional_int(
            data.get("initial_point_count"),
            field_name="latest_frontier_status.initial_point_count",
        ),
        current_best_moves_since_start=_optional_int(
            data.get("current_best_moves_since_start"),
            field_name="latest_frontier_status.current_best_moves_since_start",
        ),
        current_best_total_points=_optional_int(
            data.get("current_best_total_points"),
            field_name="latest_frontier_status.current_best_total_points",
        ),
        current_best_is_exact=_optional_bool(
            data.get("current_best_is_exact"),
            field_name="latest_frontier_status.current_best_is_exact",
        ),
        current_best_is_terminal=_optional_bool(
            data.get("current_best_is_terminal"),
            field_name="latest_frontier_status.current_best_is_terminal",
        ),
        current_best_source=_optional_str(
            data.get("current_best_source"),
            field_name="latest_frontier_status.current_best_source",
        ),
    )


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return one integer-like payload value or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapRunStateError.invalid_integer_like_value(
            field_name,
            value,
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise MalformedMorpionBootstrapRunStateError.invalid_integer_like_value(
            field_name,
            value,
        )
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise MalformedMorpionBootstrapRunStateError.invalid_integer_like_value(
                field_name,
                value,
            ) from exc
    raise MalformedMorpionBootstrapRunStateError.invalid_integer_like_value(
        field_name,
        value,
    )


def _optional_int(value: object, *, field_name: str) -> int | None:
    """Return one optional integer-like payload value or raise."""
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    """Return one optional bool payload value or raise."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise MalformedMorpionBootstrapRunStateError.invalid_bool_field(field_name)


def _optional_float(value: object, *, field_name: str) -> float | None:
    """Return one optional float-like payload value or raise."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapRunStateError.invalid_float_like_value(
            field_name,
            value,
        )
    try:
        if isinstance(value, int | float | str):
            return float(value)
    except ValueError as exc:
        raise MalformedMorpionBootstrapRunStateError.invalid_float_like_value(
            field_name,
            value,
        ) from exc
    raise MalformedMorpionBootstrapRunStateError.invalid_float_like_value(
        field_name,
        value,
    )


__all__ = [
    "MalformedMorpionBootstrapRunStateError",
    "MorpionBootstrapRunState",
    "initialize_bootstrap_run_state",
    "load_bootstrap_run_state",
    "save_bootstrap_run_state",
]
