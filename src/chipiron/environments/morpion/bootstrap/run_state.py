"""Persistence helpers for Morpion bootstrap run state."""

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
class MorpionBootstrapRunState:
    """Persisted state for a restartable Morpion bootstrap run."""

    generation: int
    latest_tree_snapshot_path: str | None
    latest_rows_path: str | None
    latest_model_bundle_path: str | None
    tree_size_at_last_save: int
    last_save_unix_s: float | None
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


def initialize_bootstrap_run_state() -> MorpionBootstrapRunState:
    """Return the initial empty run state for a fresh bootstrap run."""
    return MorpionBootstrapRunState(
        generation=0,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_path=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
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
        "latest_tree_snapshot_path": state.latest_tree_snapshot_path,
        "latest_rows_path": state.latest_rows_path,
        "latest_model_bundle_path": state.latest_model_bundle_path,
        "tree_size_at_last_save": state.tree_size_at_last_save,
        "last_save_unix_s": state.last_save_unix_s,
        "metadata": dict(state.metadata),
    }


def _run_state_from_dict(data: object) -> MorpionBootstrapRunState:
    """Deserialize one run state from JSON-friendly data."""
    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapRunStateError.invalid_top_level_mapping()

    payload = cast("Mapping[str, object]", data)
    return MorpionBootstrapRunState(
        generation=_coerce_int(payload.get("generation", 0), field_name="generation"),
        latest_tree_snapshot_path=_optional_str(
            payload.get("latest_tree_snapshot_path"),
            field_name="latest_tree_snapshot_path",
        ),
        latest_rows_path=_optional_str(
            payload.get("latest_rows_path"),
            field_name="latest_rows_path",
        ),
        latest_model_bundle_path=_optional_str(
            payload.get("latest_model_bundle_path"),
            field_name="latest_model_bundle_path",
        ),
        tree_size_at_last_save=_coerce_int(
            payload.get("tree_size_at_last_save", 0),
            field_name="tree_size_at_last_save",
        ),
        last_save_unix_s=_optional_float(
            payload.get("last_save_unix_s"),
            field_name="last_save_unix_s",
        ),
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


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return one metadata dictionary or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapRunStateError.invalid_metadata()
    return dict(cast("Mapping[str, Any]", value))


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return one integer-like payload value or raise."""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | str):
            return int(value)
        if isinstance(value, float):
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


def _optional_float(value: object, *, field_name: str) -> float | None:
    """Return one optional float-like payload value or raise."""
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(value)
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
