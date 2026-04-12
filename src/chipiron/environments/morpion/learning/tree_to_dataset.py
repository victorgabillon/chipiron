"""Convert Anemone training-tree snapshots into raw Morpion supervised rows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from anemone.training_export import load_training_tree_snapshot
from atomheart.games.morpion.checkpoints import (
    MorpionCheckpointError,
    MorpionCheckpointTypeError,
    MorpionStateCheckpointCodec,
)

MORPION_SUPERVISED_ROWS_DATASET_KIND = "morpion_supervised_rows"
MORPION_SUPERVISED_ROWS_DATASET_VERSION = 1

if TYPE_CHECKING:
    from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
    from atomheart.games.morpion.state import MorpionState as AtomMorpionState


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionSupervisedRow:
    """One raw Morpion training row derived from one exported search node."""

    node_id: str
    state_ref_payload: dict[str, Any]
    target_value: float
    is_terminal: bool
    is_exact: bool
    depth: int
    visit_count: int | None = None
    direct_value: float | None = None
    over_event_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionSupervisedRows:
    """Ordered collection of raw Morpion training rows plus extraction metadata."""

    rows: tuple[MorpionSupervisedRow, ...]
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


class InvalidMorpionStateRefPayloadError(TypeError):
    """Raised when a purported Morpion state reference payload is invalid."""

    @classmethod
    def payload_must_be_mapping(cls) -> InvalidMorpionStateRefPayloadError:
        """Return the payload-shape error."""
        return cls("Morpion state_ref_payload must be a mapping.")

    @classmethod
    def payload_keys_must_be_strings(cls) -> InvalidMorpionStateRefPayloadError:
        """Return the payload-key-type error."""
        return cls("Morpion state_ref_payload keys must all be strings.")

    @classmethod
    def payload_not_decodable(cls) -> InvalidMorpionStateRefPayloadError:
        """Return the codec-validation error."""
        return cls(
            "Morpion state_ref_payload is not decodable with "
            "MorpionStateCheckpointCodec."
        )


class MalformedMorpionSupervisedRowsError(TypeError):
    """Raised when persisted Morpion supervised rows are structurally malformed."""

    @classmethod
    def missing_rows_field(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the malformed top-level rows-field error."""
        return cls("Morpion supervised rows payload field `rows` must be a list.")

    @classmethod
    def row_entry_must_be_mapping(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the malformed row-entry error."""
        return cls("Each Morpion supervised row payload must be a dictionary.")

    @classmethod
    def missing_or_invalid_node_id(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the malformed node-id error."""
        return cls("Each Morpion supervised row must contain a string `node_id`.")

    @classmethod
    def missing_target_value(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the missing-target error."""
        return cls("Each Morpion supervised row must contain a numeric `target_value`.")


def is_morpion_state_ref_payload(payload: object) -> bool:
    """Return whether ``payload`` is a decodable Morpion checkpoint payload."""
    try:
        _validate_and_normalize_state_ref_payload(payload)
    except InvalidMorpionStateRefPayloadError:
        return False
    return True


def decode_morpion_state_ref_payload(
    payload: Mapping[str, object],
) -> AtomMorpionState:
    """Decode one validated Morpion checkpoint payload into an atomheart state."""
    normalized_payload = _validate_and_normalize_state_ref_payload(payload)
    return _decode_validated_payload(normalized_payload)


def training_node_to_morpion_supervised_row(
    node: TrainingNodeSnapshot,
    *,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    use_backed_up_value: bool = True,
) -> MorpionSupervisedRow | None:
    """Convert one exported training node into one raw Morpion supervised row.

    ``target_value`` is derived from the exported node scalar. By default this
    uses ``backed_up_value_scalar`` because later training will bootstrap from
    tree backups. Terminal or exact nodes are higher-confidence targets, but
    this raw-row format does not attach a separate confidence weight yet.
    """
    if node.state_ref_payload is None:
        return None

    target_value = _choose_target_value(node, use_backed_up_value=use_backed_up_value)
    if target_value is None:
        return None

    if not _passes_filters(
        node,
        require_exact_or_terminal=require_exact_or_terminal,
        min_depth=min_depth,
        min_visit_count=min_visit_count,
    ):
        return None

    normalized_payload = _validate_and_normalize_state_ref_payload(
        node.state_ref_payload
    )

    return MorpionSupervisedRow(
        node_id=node.node_id,
        state_ref_payload=normalized_payload,
        target_value=target_value,
        is_terminal=node.is_terminal,
        is_exact=node.is_exact,
        depth=node.depth,
        visit_count=node.visit_count,
        direct_value=node.direct_value_scalar,
        over_event_label=node.over_event_label,
        metadata=dict(node.metadata),
    )


def training_tree_snapshot_to_morpion_supervised_rows(
    snapshot: TrainingTreeSnapshot,
    *,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    max_rows: int | None = None,
    use_backed_up_value: bool = True,
    metadata: dict[str, object] | None = None,
) -> MorpionSupervisedRows:
    """Extract ordered raw Morpion supervised rows from one training snapshot."""
    rows = tuple(
        row
        for row in (
            training_node_to_morpion_supervised_row(
                node,
                require_exact_or_terminal=require_exact_or_terminal,
                min_depth=min_depth,
                min_visit_count=min_visit_count,
                use_backed_up_value=use_backed_up_value,
            )
            for node in snapshot.nodes
        )
        if row is not None
    )
    if max_rows is not None:
        rows = rows[:max_rows]
    return MorpionSupervisedRows(
        rows=rows,
        metadata=_build_rows_metadata(
            snapshot,
            metadata=metadata,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
            max_rows=max_rows,
            use_backed_up_value=use_backed_up_value,
            num_rows=len(rows),
        ),
    )


def load_training_tree_snapshot_as_morpion_supervised_rows(
    path: str | Path,
    *,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    max_rows: int | None = None,
    use_backed_up_value: bool = True,
    metadata: dict[str, object] | None = None,
) -> MorpionSupervisedRows:
    """Load one persisted Anemone training snapshot and extract Morpion rows."""
    snapshot = load_training_tree_snapshot(path)
    return training_tree_snapshot_to_morpion_supervised_rows(
        snapshot,
        require_exact_or_terminal=require_exact_or_terminal,
        min_depth=min_depth,
        min_visit_count=min_visit_count,
        max_rows=max_rows,
        use_backed_up_value=use_backed_up_value,
        metadata=metadata,
    )


def morpion_supervised_rows_to_dict(data: MorpionSupervisedRows) -> dict[str, object]:
    """Serialize raw Morpion supervised rows into a JSON-friendly dictionary."""
    return {
        "rows": [_row_to_dict(row) for row in data.rows],
        "metadata": dict(data.metadata),
    }


def morpion_supervised_rows_from_dict(
    data: dict[str, object],
) -> MorpionSupervisedRows:
    """Deserialize raw Morpion supervised rows from JSON-friendly data."""
    rows_data = data.get("rows")
    if not isinstance(rows_data, list):
        raise MalformedMorpionSupervisedRowsError.missing_rows_field()
    typed_rows_data = cast("list[object]", rows_data)

    return MorpionSupervisedRows(
        rows=tuple(
            _row_from_dict(_require_row_mapping(item)) for item in typed_rows_data
        ),
        metadata=_metadata_dict(data.get("metadata")),
    )


def save_morpion_supervised_rows(
    data: MorpionSupervisedRows,
    path: str | Path,
) -> None:
    """Persist raw Morpion supervised rows as UTF-8 JSON.

    The saved payload is JSON, so row contents, especially ``state_ref_payload``,
    must already be JSON-serializable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(morpion_supervised_rows_to_dict(data), indent=2) + "\n",
        encoding="utf-8",
    )


def load_morpion_supervised_rows(
    path: str | Path,
) -> MorpionSupervisedRows:
    """Load raw Morpion supervised rows from ``path``."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return morpion_supervised_rows_from_dict(cast("dict[str, object]", loaded))


def _validate_and_normalize_state_ref_payload(payload: object) -> dict[str, Any]:
    """Return a normalized Morpion checkpoint payload after codec validation."""
    normalized_payload = _payload_mapping(payload)
    _decode_validated_payload(normalized_payload)
    return normalized_payload


def _decode_validated_payload(payload: dict[str, Any]) -> AtomMorpionState:
    """Decode one normalized checkpoint payload or raise a Morpion payload error."""
    return _load_morpion_state_from_payload(payload)


def _payload_mapping(payload: object) -> dict[str, Any]:
    """Return a string-keyed payload mapping or raise."""
    if not isinstance(payload, Mapping):
        raise InvalidMorpionStateRefPayloadError.payload_must_be_mapping()

    raw_payload = cast("Mapping[object, object]", payload)
    normalized_payload: dict[str, Any] = {}
    for key_obj, value_obj in raw_payload.items():
        if not isinstance(key_obj, str):
            raise InvalidMorpionStateRefPayloadError.payload_keys_must_be_strings()
        normalized_payload[key_obj] = value_obj
    return normalized_payload


def _choose_target_value(
    node: TrainingNodeSnapshot,
    *,
    use_backed_up_value: bool,
) -> float | None:
    """Return the preferred target scalar with no fallback to the other field."""
    if use_backed_up_value:
        return node.backed_up_value_scalar
    return node.direct_value_scalar


def _load_morpion_state_from_payload(payload: dict[str, Any]) -> AtomMorpionState:
    """Load one Morpion state from an already normalized checkpoint payload."""
    try:
        return MorpionStateCheckpointCodec().load_state_ref(payload)
    except (MorpionCheckpointError, MorpionCheckpointTypeError) as exc:
        raise InvalidMorpionStateRefPayloadError.payload_not_decodable() from exc


def _passes_filters(
    node: TrainingNodeSnapshot,
    *,
    require_exact_or_terminal: bool,
    min_depth: int | None,
    min_visit_count: int | None,
) -> bool:
    """Return whether one exported node passes extraction filters."""
    if require_exact_or_terminal and not (node.is_exact or node.is_terminal):
        return False
    if min_depth is not None and node.depth < min_depth:
        return False
    if min_visit_count is not None:
        if node.visit_count is None:
            return False
        if node.visit_count < min_visit_count:
            return False
    return True


def _build_rows_metadata(
    snapshot: TrainingTreeSnapshot,
    *,
    metadata: dict[str, object] | None,
    require_exact_or_terminal: bool,
    min_depth: int | None,
    min_visit_count: int | None,
    max_rows: int | None,
    use_backed_up_value: bool,
    num_rows: int,
) -> dict[str, Any]:
    """Build dataset metadata for one extraction pass."""
    built_metadata: dict[str, Any] = {
        "dataset_kind": MORPION_SUPERVISED_ROWS_DATASET_KIND,
        "dataset_version": MORPION_SUPERVISED_ROWS_DATASET_VERSION,
        "source_root_node_id": snapshot.root_node_id,
        "source_format_kind": snapshot.metadata.get("format_kind"),
        "source_format_version": snapshot.metadata.get("format_version"),
        "require_exact_or_terminal": require_exact_or_terminal,
        "min_depth": min_depth,
        "min_visit_count": min_visit_count,
        "max_rows": max_rows,
        "use_backed_up_value": use_backed_up_value,
        "num_rows": num_rows,
    }
    if metadata is not None:
        built_metadata.update(metadata)
    return built_metadata


def _row_to_dict(row: MorpionSupervisedRow) -> dict[str, object]:
    """Serialize one Morpion supervised row to JSON-friendly data."""
    return {
        "node_id": row.node_id,
        "state_ref_payload": dict(row.state_ref_payload),
        "target_value": row.target_value,
        "is_terminal": row.is_terminal,
        "is_exact": row.is_exact,
        "depth": row.depth,
        "visit_count": row.visit_count,
        "direct_value": row.direct_value,
        "over_event_label": row.over_event_label,
        "metadata": dict(row.metadata),
    }


def _require_row_mapping(value: object) -> dict[str, object]:
    """Return one row payload or raise for malformed entries."""
    if not isinstance(value, dict):
        raise MalformedMorpionSupervisedRowsError.row_entry_must_be_mapping()
    return cast("dict[str, object]", value)


def _row_from_dict(data: dict[str, object]) -> MorpionSupervisedRow:
    """Deserialize one Morpion supervised row from JSON-friendly data."""
    node_id = data.get("node_id")
    if not isinstance(node_id, str):
        raise MalformedMorpionSupervisedRowsError.missing_or_invalid_node_id()

    if "target_value" not in data:
        raise MalformedMorpionSupervisedRowsError.missing_target_value()

    return MorpionSupervisedRow(
        node_id=node_id,
        state_ref_payload=_validate_and_normalize_state_ref_payload(
            data.get("state_ref_payload")
        ),
        target_value=_required_float(data["target_value"]),
        is_terminal=bool(data.get("is_terminal", False)),
        is_exact=bool(data.get("is_exact", False)),
        depth=_coerce_int(data.get("depth", 0), default=0),
        visit_count=_optional_int(data.get("visit_count")),
        direct_value=_optional_float(data.get("direct_value")),
        over_event_label=_optional_str(data.get("over_event_label")),
        metadata=_metadata_dict(data.get("metadata")),
    )


def _required_float(value: object) -> float:
    """Return one numeric value or raise for a missing target."""
    loaded = _optional_float(value)
    if loaded is None:
        raise MalformedMorpionSupervisedRowsError.missing_target_value()
    return loaded


def _optional_float(value: object) -> float | None:
    """Return ``value`` as ``float`` unless it is ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError


def _optional_int(value: object) -> int | None:
    """Return ``value`` as ``int`` unless it is ``None``."""
    if value is None:
        return None
    return _coerce_int(value)


def _optional_str(value: object) -> str | None:
    """Return ``value`` as ``str`` unless it is ``None``."""
    return None if value is None else str(value)


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return a shallow-copied metadata dictionary when possible."""
    if not isinstance(value, dict):
        return {}
    return dict(cast("dict[str, Any]", value))


def _coerce_int(value: object, *, default: int | None = None) -> int:
    """Return ``value`` as ``int`` for supported scalar payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if default is not None:
        return default
    raise TypeError


__all__ = [
    "MORPION_SUPERVISED_ROWS_DATASET_KIND",
    "MORPION_SUPERVISED_ROWS_DATASET_VERSION",
    "InvalidMorpionStateRefPayloadError",
    "MalformedMorpionSupervisedRowsError",
    "MorpionSupervisedRow",
    "MorpionSupervisedRows",
    "decode_morpion_state_ref_payload",
    "is_morpion_state_ref_payload",
    "load_morpion_supervised_rows",
    "load_training_tree_snapshot_as_morpion_supervised_rows",
    "morpion_supervised_rows_from_dict",
    "morpion_supervised_rows_to_dict",
    "save_morpion_supervised_rows",
    "training_node_to_morpion_supervised_row",
    "training_tree_snapshot_to_morpion_supervised_rows",
]
