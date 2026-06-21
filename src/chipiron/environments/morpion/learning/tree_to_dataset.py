"""Convert Anemone training-tree snapshots into raw Morpion supervised rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from anemone.training_export import load_training_tree_snapshot
from atomheart.games.morpion.checkpoints import (
    MorpionCheckpointError,
    MorpionCheckpointTypeError,
    MorpionStateCheckpointCodec,
)

MORPION_SUPERVISED_ROWS_DATASET_KIND = "morpion_supervised_rows"
MORPION_SUPERVISED_ROWS_DATASET_VERSION = 1
MORPION_SUPERVISED_ROWS_JSONL_FORMAT_KIND = "morpion_supervised_rows_jsonl"
MORPION_SUPERVISED_ROWS_JSONL_FORMAT_VERSION = 1
MORPION_SUPERVISED_ROWS_METADATA_RECORD_KIND = "morpion_supervised_rows_metadata"
MORPION_SUPERVISED_ROW_RECORD_KIND = "morpion_supervised_row"

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


@dataclass(frozen=True, slots=True)
class MorpionSupervisedRowsWriteStats:
    """Stats returned after streaming supervised rows to disk."""

    row_count: int
    bytes_written: int
    path: Path
    format_kind: str
    format_version: int


@dataclass(frozen=True, slots=True)
class MorpionSupervisedRowsSource:
    """Format-aware metadata for one supervised rows artifact."""

    path: Path
    metadata: dict[str, Any]
    row_count: int | None
    format_kind: Literal["json", "jsonl"]


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

    @classmethod
    def unexpected_jsonl_record_kind(
        cls,
        line_number: int,
        kind: object,
    ) -> MalformedMorpionSupervisedRowsError:
        """Return the malformed JSONL record-kind error."""
        return cls(
            f"Unexpected Morpion supervised rows JSONL record kind "
            f"at line {line_number}: {kind!r}."
        )

    @classmethod
    def missing_jsonl_metadata_record(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the missing JSONL metadata-record error."""
        return cls("Morpion supervised rows JSONL is missing its metadata record.")

    @classmethod
    def invalid_chunk_size(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the invalid streaming chunk-size error."""
        return cls("Morpion supervised row chunk_size must be a positive integer.")

    @classmethod
    def invalid_max_rows(cls) -> MalformedMorpionSupervisedRowsError:
        """Return the invalid streaming row-limit error."""
        return cls("Morpion supervised max_rows must be a non-negative integer or None.")


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

    target = _choose_target_value_and_source(
        node,
        use_backed_up_value=use_backed_up_value,
    )
    if target is None:
        return None
    target_value, target_source = target

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
        metadata={**dict(node.metadata), "target_source": target_source},
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
        iter_morpion_supervised_rows_from_training_snapshot(
            snapshot,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
            max_rows=max_rows,
            use_backed_up_value=use_backed_up_value,
        )
    )
    return MorpionSupervisedRows(
        rows=rows,
        metadata=morpion_supervised_rows_metadata_from_training_snapshot(
            snapshot,
            metadata=metadata,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
            max_rows=max_rows,
            use_backed_up_value=use_backed_up_value,
        ),
    )


def iter_morpion_supervised_rows_from_training_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    max_rows: int | None = None,
    use_backed_up_value: bool = True,
) -> Iterable[MorpionSupervisedRow]:
    """Yield ordered raw Morpion supervised rows from one training snapshot."""
    emitted_count = 0
    for node in snapshot.nodes:
        row = training_node_to_morpion_supervised_row(
            node,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
            use_backed_up_value=use_backed_up_value,
        )
        if row is None:
            continue
        if max_rows is not None and emitted_count >= max_rows:
            break
        emitted_count += 1
        yield row


def morpion_supervised_rows_metadata_from_training_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    max_rows: int | None = None,
    use_backed_up_value: bool = True,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Build extraction metadata without materializing supervised rows."""
    stats = _collect_extraction_stats(
        snapshot,
        require_exact_or_terminal=require_exact_or_terminal,
        min_depth=min_depth,
        min_visit_count=min_visit_count,
        max_rows=max_rows,
        use_backed_up_value=use_backed_up_value,
    )
    return _build_rows_metadata(
        snapshot,
        metadata=metadata,
        require_exact_or_terminal=require_exact_or_terminal,
        min_depth=min_depth,
        min_visit_count=min_visit_count,
        max_rows=max_rows,
        use_backed_up_value=use_backed_up_value,
        num_rows=stats.num_rows,
        skipped_no_target_count=stats.skipped_no_target_count,
        target_source_counts=stats.target_source_counts,
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


def save_morpion_supervised_rows_streaming(
    *,
    rows: Iterable[MorpionSupervisedRow],
    metadata: Mapping[str, object],
    path: str | Path,
    progress_callback: Callable[[int], None] | None = None,
    progress_interval: int = 10_000,
) -> MorpionSupervisedRowsWriteStats:
    """Stream raw Morpion supervised rows as atomic UTF-8 JSON Lines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    row_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            _write_jsonl_record(
                stream,
                {
                    "kind": MORPION_SUPERVISED_ROWS_METADATA_RECORD_KIND,
                    "format_version": MORPION_SUPERVISED_ROWS_JSONL_FORMAT_VERSION,
                    "metadata": dict(metadata),
                },
            )
            for row in rows:
                _write_jsonl_record(
                    stream,
                    {
                        "kind": MORPION_SUPERVISED_ROW_RECORD_KIND,
                        "row": _row_to_dict(row),
                    },
                )
                row_count += 1
                if (
                    progress_callback is not None
                    and progress_interval > 0
                    and row_count % progress_interval == 0
                ):
                    progress_callback(row_count)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return MorpionSupervisedRowsWriteStats(
        row_count=row_count,
        bytes_written=target.stat().st_size,
        path=target,
        format_kind=MORPION_SUPERVISED_ROWS_JSONL_FORMAT_KIND,
        format_version=MORPION_SUPERVISED_ROWS_JSONL_FORMAT_VERSION,
    )


def load_morpion_supervised_rows_metadata(path: str | Path) -> dict[str, Any]:
    """Load only metadata from a supervised rows artifact when possible."""
    source = Path(path)
    if source.suffix != ".jsonl":
        return load_morpion_supervised_rows(source).metadata
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise MalformedMorpionSupervisedRowsError.row_entry_must_be_mapping()
            kind = record.get("kind")
            if kind != MORPION_SUPERVISED_ROWS_METADATA_RECORD_KIND:
                raise MalformedMorpionSupervisedRowsError.unexpected_jsonl_record_kind(
                    line_number,
                    kind,
                )
            return _metadata_dict(record.get("metadata"))
    raise MalformedMorpionSupervisedRowsError.missing_jsonl_metadata_record()


def iter_morpion_supervised_rows_from_path(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[MorpionSupervisedRow]:
    """Yield rows from old JSON or new JSONL artifacts."""
    _validate_optional_max_rows(max_rows)
    source = Path(path)
    emitted = 0
    if source.suffix != ".jsonl":
        for row in load_morpion_supervised_rows(source).rows:
            if max_rows is not None and emitted >= max_rows:
                break
            emitted += 1
            yield row
        return
    saw_metadata = False
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise MalformedMorpionSupervisedRowsError.row_entry_must_be_mapping()
            kind = record.get("kind")
            if kind == MORPION_SUPERVISED_ROWS_METADATA_RECORD_KIND:
                saw_metadata = True
                continue
            if not saw_metadata:
                raise MalformedMorpionSupervisedRowsError.missing_jsonl_metadata_record()
            if kind != MORPION_SUPERVISED_ROW_RECORD_KIND:
                raise MalformedMorpionSupervisedRowsError.unexpected_jsonl_record_kind(
                    line_number,
                    kind,
                )
            if max_rows is not None and emitted >= max_rows:
                break
            emitted += 1
            yield _row_from_dict(_require_row_mapping(record.get("row")))
    if not saw_metadata:
        raise MalformedMorpionSupervisedRowsError.missing_jsonl_metadata_record()


def iter_morpion_supervised_row_chunks_from_path(
    path: str | Path,
    *,
    chunk_size: int,
    max_rows: int | None = None,
) -> Iterator[tuple[MorpionSupervisedRow, ...]]:
    """Yield non-empty chunks of rows from one supervised rows artifact."""
    if isinstance(chunk_size, bool) or chunk_size <= 0:
        raise MalformedMorpionSupervisedRowsError.invalid_chunk_size()
    chunk: list[MorpionSupervisedRow] = []
    for row in iter_morpion_supervised_rows_from_path(path, max_rows=max_rows):
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def morpion_supervised_rows_source_from_path(
    path: str | Path,
) -> MorpionSupervisedRowsSource:
    """Return format-aware metadata for one supervised rows artifact."""
    source = Path(path)
    metadata = load_morpion_supervised_rows_metadata(source)
    if source.suffix == ".jsonl":
        return MorpionSupervisedRowsSource(
            path=source,
            metadata=metadata,
            row_count=_metadata_optional_int(metadata.get("num_rows")),
            format_kind="jsonl",
        )
    rows = load_morpion_supervised_rows(source)
    return MorpionSupervisedRowsSource(
        path=source,
        metadata=rows.metadata,
        row_count=len(rows.rows),
        format_kind="json",
    )


def load_morpion_supervised_rows(
    path: str | Path,
) -> MorpionSupervisedRows:
    """Load raw Morpion supervised rows from ``path``."""
    source = Path(path)
    if source.suffix == ".jsonl":
        return _load_morpion_supervised_rows_jsonl(source)
    loaded = json.loads(source.read_text(encoding="utf-8"))
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


def _choose_target_value_and_source(
    node: TrainingNodeSnapshot,
    *,
    use_backed_up_value: bool,
) -> tuple[float, str] | None:
    """Return the preferred target scalar together with its provenance label."""
    if use_backed_up_value and node.backed_up_value_scalar is not None:
        return float(node.backed_up_value_scalar), "backed_up_value"
    if (node.is_exact or node.is_terminal) and node.direct_value_scalar is not None:
        return float(node.direct_value_scalar), "exact_or_terminal_direct_value"
    if not use_backed_up_value and node.direct_value_scalar is not None:
        return float(node.direct_value_scalar), "direct_value_frontier_fallback"
    if not use_backed_up_value and node.backed_up_value_scalar is not None:
        return float(node.backed_up_value_scalar), "backed_up_value"
    return None


@dataclass(frozen=True, slots=True)
class _ExtractionStats:
    num_rows: int
    skipped_no_target_count: int
    target_source_counts: dict[str, int]


def _collect_extraction_stats(
    snapshot: TrainingTreeSnapshot,
    *,
    require_exact_or_terminal: bool,
    min_depth: int | None,
    min_visit_count: int | None,
    max_rows: int | None,
    use_backed_up_value: bool,
) -> _ExtractionStats:
    """Return compact extraction stats without building row objects."""
    skipped_no_target_count = 0
    num_rows = 0
    target_source_counts = _empty_target_source_counts()
    for node in snapshot.nodes:
        if node.state_ref_payload is None:
            continue
        if not _passes_filters(
            node,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
        ):
            continue
        target = _choose_target_value_and_source(
            node,
            use_backed_up_value=use_backed_up_value,
        )
        if target is None:
            skipped_no_target_count += 1
            continue
        if max_rows is not None and num_rows >= max_rows:
            continue
        _target_value, target_source = target
        if target_source in target_source_counts:
            target_source_counts[target_source] += 1
        num_rows += 1
    return _ExtractionStats(
        num_rows=num_rows,
        skipped_no_target_count=skipped_no_target_count,
        target_source_counts=target_source_counts,
    )


def _target_source_counts(
    rows: tuple[MorpionSupervisedRow, ...],
) -> dict[str, int]:
    """Return compact per-source counts for the extracted rows."""
    counts = _empty_target_source_counts()
    for row in rows:
        source = row.metadata.get("target_source")
        if isinstance(source, str) and source in counts:
            counts[source] += 1
    return counts


def _empty_target_source_counts() -> dict[str, int]:
    return {
        "backed_up_value": 0,
        "exact_or_terminal_direct_value": 0,
        "direct_value_frontier_fallback": 0,
    }


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
    skipped_no_target_count: int,
    target_source_counts: dict[str, int],
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
        "skipped_no_target_count": skipped_no_target_count,
        "target_source_counts": dict(target_source_counts),
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


def _write_jsonl_record(stream: Any, record: Mapping[str, object]) -> None:
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _load_morpion_supervised_rows_jsonl(path: Path) -> MorpionSupervisedRows:
    metadata = load_morpion_supervised_rows_metadata(path)
    rows = list(iter_morpion_supervised_rows_from_path(path))
    return MorpionSupervisedRows(rows=tuple(rows), metadata=metadata)


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


def _metadata_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _validate_optional_max_rows(max_rows: int | None) -> None:
    if max_rows is None:
        return
    if isinstance(max_rows, bool) or max_rows < 0:
        raise MalformedMorpionSupervisedRowsError.invalid_max_rows()


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
    "MORPION_SUPERVISED_ROWS_JSONL_FORMAT_KIND",
    "MORPION_SUPERVISED_ROWS_JSONL_FORMAT_VERSION",
    "InvalidMorpionStateRefPayloadError",
    "MalformedMorpionSupervisedRowsError",
    "MorpionSupervisedRow",
    "MorpionSupervisedRows",
    "MorpionSupervisedRowsSource",
    "MorpionSupervisedRowsWriteStats",
    "decode_morpion_state_ref_payload",
    "is_morpion_state_ref_payload",
    "iter_morpion_supervised_row_chunks_from_path",
    "iter_morpion_supervised_rows_from_path",
    "iter_morpion_supervised_rows_from_training_snapshot",
    "load_morpion_supervised_rows",
    "load_morpion_supervised_rows_metadata",
    "load_training_tree_snapshot_as_morpion_supervised_rows",
    "morpion_supervised_rows_from_dict",
    "morpion_supervised_rows_metadata_from_training_snapshot",
    "morpion_supervised_rows_source_from_path",
    "morpion_supervised_rows_to_dict",
    "save_morpion_supervised_rows",
    "save_morpion_supervised_rows_streaming",
    "training_node_to_morpion_supervised_row",
    "training_tree_snapshot_to_morpion_supervised_rows",
]
