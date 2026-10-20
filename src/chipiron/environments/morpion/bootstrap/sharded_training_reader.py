"""Read published sharded training exports with bounded RAM and disposable SQLite.

Only individual JSON rows are built. No Morpion states or tree snapshots are
constructed during indexing. Memory is O(window + largest row + parser buffer +
SQLite cache + generation manifests); disk is O(all immutable rows + G updates).
"""
# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager, suppress
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

import ijson
from platformdirs import user_cache_path

from .performance import StageMeasurement
from .pipeline_memory import log_pipeline_memory
from .sharded_training_export import (
    MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
    MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
    _load_generation_manifest,
    _load_root_manifest,
    _merge_node_record_and_update,
    _node_record_from_dict,
    _node_record_to_dict,
    _node_update_from_dict,
    _node_update_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    from anemone.training_export import TrainingNodeSnapshot

LOGGER = logging.getLogger(__name__)
INDEX_SCHEMA_VERSION = 1
PARSER_BUFFER_BYTES = 64 * 1024
SQLITE_CACHE_KIB = 2048
TRANSACTION_ROWS = 256
MANIFEST_MAX_BYTES = 4 * 1024 * 1024

type WindowSelection = tuple[tuple[str, ...], str | None, bool]


def _has_surrogate_escape(path: Path) -> bool:
    """Conservatively choose a lossless library backend, without parsing JSON.

    YAJL replaces unpaired high-surrogate escapes with '?'. Detect all possible
    surrogate escapes, including escaped backslashes, across fixed read chunks.
    False positives only select a slower backend; they never change values.
    """
    found = False
    previous = b""
    with path.open("rb") as stream:
        while chunk := stream.read(PARSER_BUFFER_BYTES):
            data = previous + chunk
            found = found or re.search(rb"\\u[dD][89a-fA-F]", data) is not None
            previous = data[-5:]
    return found


def _shard_events(path: Path, field: str) -> Iterator[tuple[str, str, object]]:
    """Validate the array envelope using parser events, normalizing JSON floats.

    Decimal-to-float conversion matches json.loads without the large-integer
    overflow of YAJL's use_float=True. ijson builds only one item at a time.
    Missing/wrong arrays and duplicate top-level keys fail explicitly, rather
    than looking like empty populations. All input, including the tail, is read.
    """
    # The Python backend uses stdlib scanstring and preserves surrogate escapes.
    # Its lexer can retain consumed prefixes when a string spans every chunk.
    # One-byte input guarantees a discard boundary after every completed token.
    # Only uncommon surrogate-escaped shards need this slower cold-build path.
    lossless_unicode = _has_surrogate_escape(path) or ijson.backend == "python"
    keys: set[str] = set()
    found_array = False
    with path.open("rb") as stream:
        if lossless_unicode:
            events = ijson.get_backend("python").parse(stream, buf_size=1)
        else:
            events = ijson.parse(stream, buf_size=PARSER_BUFFER_BYTES)
        if next(events, None) != ("", "start_map", None):
            raise ValueError(f"Expected a JSON object in {path}")
        yield "", "start_map", None
        for prefix, event, value in events:
            if prefix == "" and event == "map_key":
                key = str(value)
                if key in keys:
                    raise ValueError(f"Duplicate top-level key {key!r} in {path}")
                keys.add(key)
                if len(keys) > 64:
                    raise ValueError(f"Too many top-level shard keys in {path}")
            if prefix == field and not found_array:
                if event != "start_array":
                    raise ValueError(f"Expected {field!r} array in {path}")
                found_array = True
            if (
                prefix == "format_kind"
                and event == "string"
                and value != MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND
            ):
                raise ValueError(f"Unsupported shard format in {path}")
            if (
                prefix == "format_version"
                and event == "number"
                and value != MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION
            ):
                raise ValueError(f"Unsupported shard version in {path}")
            yield prefix, event, float(value) if isinstance(value, Decimal) else value
    if not found_array:
        raise ValueError(f"Missing {field!r} array in {path}")


def iter_shard_rows(path: Path, field: str) -> Iterator[dict[str, object]]:
    """Yield validated JSON array records, retaining no complete input array."""
    try:
        for row in ijson.items(_shard_events(path, field), f"{field}.item"):
            if not isinstance(row, dict):
                raise TypeError(f"Expected object in {path} {field!r} array")
            yield cast("dict[str, object]", row)
    except ijson.JSONError as exc:
        raise ValueError(f"Malformed or incomplete JSON shard {path}: {exc}") from exc


def _read_manifest_identity(path: Path) -> dict[str, object]:
    """Read a small manifest; refuse unbounded metadata in bounded mode."""
    if path.stat().st_size > MANIFEST_MAX_BYTES:
        raise ValueError(f"Manifest exceeds bounded-reader size limit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected manifest object in {path}")
    if (
        payload.get("format_kind", MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND)
        != MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND
    ):
        raise ValueError(f"Unsupported manifest format in {path}")
    if (
        payload.get("format_version", 1)
        != MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported manifest version in {path}")
    return cast("dict[str, object]", payload)


def is_sharded_generation_manifest(path: Path) -> bool:
    """Recognize small generation manifests without reading flat snapshot arrays."""
    if path.stat().st_size > MANIFEST_MAX_BYTES:
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and all(
        key in payload for key in ("node_shard_path", "update_shard_path", "generation")
    )


def _configure_connection(connection: sqlite3.Connection) -> None:
    """Bound page cache and sorting RAM; never mmap the whole index."""
    connection.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KIB}")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")


def _validated_node_id(node_id: str) -> str:
    """Require Unicode scalar IDs supported by SQLite UTF-8 BINARY collation.

    UTF-8 byte order preserves Unicode scalar/code-point order, including NUL,
    non-ASCII and astral characters. Lone surrogates are rejected explicitly.
    """
    try:
        node_id.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Bounded reader requires Unicode scalar node IDs") from exc
    return node_id


class ShardedTrainingGenerationView:
    """Context-managed indexed view of exactly one published export generation.

    Cache files live outside source artifacts. Every open fingerprints all
    relevant files in fixed buffers, including on reuse; no stat-only trust.
    New indexes are committed/validated in a unique temporary file and renamed
    only when complete. Concurrent builders may safely publish identical views.
    """

    def __init__(
        self,
        generation_manifest_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize/reuse an index without constructing training snapshots."""
        measurement = StageMeasurement()
        self.source_bytes_read = 0
        self.index_build_peak_disk_bytes = 0
        self.manifest_path = Path(generation_manifest_path).resolve()
        self.root = self.manifest_path.parent
        _read_manifest_identity(self.manifest_path)
        self.manifest = _load_generation_manifest(self.manifest_path)
        self.source_bytes_read += 2 * self.manifest_path.stat().st_size
        if self.manifest.generation < 1:
            raise ValueError("Sharded export generation must be positive")
        self.root_node_id = self.manifest.root_node_id
        self.generation = self.manifest.generation
        self._node_shards: list[Path] = []
        self._source_paths: list[Path] = [self.manifest_path]
        root_manifest_path = self.root / "manifest.json"
        if root_manifest_path.exists():
            _read_manifest_identity(root_manifest_path)
            self._source_paths.append(root_manifest_path)
        root_manifest = _load_root_manifest(root_manifest_path)
        if root_manifest_path.is_file():
            self.source_bytes_read += 2 * root_manifest_path.stat().st_size
        for generation in range(1, self.generation + 1):
            relative = root_manifest.generation_manifests.get(str(generation))
            if relative is None:
                continue  # Same referenced-generation selection as the full loader.
            path = self.root / relative
            _read_manifest_identity(path)
            shard_manifest = _load_generation_manifest(path)
            self.source_bytes_read += 2 * path.stat().st_size
            self._source_paths.append(path)
            shard = self.root / shard_manifest.node_shard_path
            self._source_paths.append(shard)
            self._node_shards.append(shard)
        self._update_shard = self.root / self.manifest.update_shard_path
        self._source_paths.append(self._update_shard)
        self._source_paths = list(dict.fromkeys(self._source_paths))
        self.identity = self._source_identity()
        directory = (
            Path(cache_dir)
            if cache_dir is not None
            else user_cache_path("chipiron") / "sharded_training_reader"
        ).resolve()
        # Even an explicit override must never place a cache inside the export.
        if directory == self.root or self.root in directory.parents:
            raise ValueError("Reader cache must be outside the source export directory")
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        self.index_path = directory / f"v{INDEX_SCHEMA_VERSION}-{digest}.sqlite3"
        index_measurement = StageMeasurement()
        self.index_reused = self._valid_index()
        if not self.index_reused:
            self._build_index()
        self.index_observation = index_measurement.finish(
            status="reused" if self.index_reused else "built",
        )
        self._connection = sqlite3.connect(
            f"{self.index_path.as_uri()}?mode=ro", uri=True
        )
        _configure_connection(self._connection)
        self._count = int(
            self._connection.execute("SELECT COUNT(*) FROM updates").fetchone()[0]
        )
        self.index_bytes = self.index_path.stat().st_size
        self.initialization_observation = measurement.finish(
            reader_mode="sharded_indexed",
            source_generation=self.generation,
            source_node_count=self._count,
            index_reused=self.index_reused,
            index_bytes=self.index_bytes,
            index_build_peak_disk_bytes=self.index_build_peak_disk_bytes,
            source_bytes_read=self.source_bytes_read,
            index=self.index_observation,
        )
        LOGGER.info("[sharded-reader] %s", json.dumps(self.initialization_observation))
        log_pipeline_memory(
            stage="reevaluation",
            generation=self.generation,
            event="reader_ready",
            reader_mode="sharded_indexed",
            node_count=self._count,
            index_reused=self.index_reused,
            index_bytes=self.index_bytes,
        )

    def _source_identity(self) -> str:
        """Hash exact source contents with a fixed-size buffer on every open."""
        sources: list[dict[str, object]] = []
        for path in self._source_paths:
            before = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.source_bytes_read += before.st_size
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(
                    f"Training export changed during fingerprinting: {path}"
                )
            sources.append({
                "path": str(path.resolve()),
                "size": before.st_size,
                "sha256": digest,
            })
        return json.dumps(
            {
                "schema": INDEX_SCHEMA_VERSION,
                "format": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
                "manifest": str(self.manifest_path),
                "sources": sources,
            },
            sort_keys=True,
        )

    def _valid_index(self) -> bool:
        """Accept only completed caches with the exact source/schema identity."""
        if not self.index_path.is_file():
            return False
        try:
            connection = sqlite3.connect(
                f"{self.index_path.as_uri()}?mode=ro", uri=True
            )
            try:
                _configure_connection(connection)
                row = connection.execute("SELECT identity FROM completed").fetchone()
                return bool(
                    row == (self.identity,)
                    and connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
                )
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            return False

    def _build_index(self) -> None:
        """Stream records in bounded transactions and atomically publish a cache."""
        descriptor, name = tempfile.mkstemp(
            prefix=".building-", suffix=".sqlite3", dir=self.index_path.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        connection = sqlite3.connect(temporary)
        try:
            _configure_connection(connection)
            connection.executescript("""
                CREATE TABLE records (
                    node_id TEXT COLLATE BINARY PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE updates (
                    node_id TEXT COLLATE BINARY PRIMARY KEY REFERENCES records(node_id),
                    order_index INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX export_order ON updates(order_index, sequence);
                CREATE TABLE completed (identity TEXT NOT NULL);
            """)
            count = 0
            for path in self._node_shards:
                for row in iter_shard_rows(path, "nodes"):
                    record = _node_record_from_dict(row)
                    connection.execute(
                        "INSERT INTO records VALUES (?, ?)",
                        (
                            _validated_node_id(record.node_id),
                            json.dumps(_node_record_to_dict(record)),
                        ),
                    )
                    count += 1
                    if count % TRANSACTION_ROWS == 0:
                        self._measure_build_disk(temporary)
                        connection.commit()
                self.source_bytes_read += 2 * path.stat().st_size
            for sequence, row in enumerate(
                iter_shard_rows(self._update_shard, "updates")
            ):
                update = _node_update_from_dict(row)
                if not -(2**63) <= update.order_index < 2**63:
                    raise ValueError("order_index exceeds SQLite signed 64-bit range")
                connection.execute(
                    "INSERT INTO updates VALUES (?, ?, ?, ?)",
                    (
                        _validated_node_id(update.node_id),
                        update.order_index,
                        sequence,
                        json.dumps(_node_update_to_dict(update)),
                    ),
                )
                if (sequence + 1) % TRANSACTION_ROWS == 0:
                    self._measure_build_disk(temporary)
                    connection.commit()
            self.source_bytes_read += 2 * self._update_shard.stat().st_size
            count = connection.execute("SELECT COUNT(*) FROM updates").fetchone()[0]
            if count != self.manifest.node_count:
                raise ValueError(
                    f"Generation node_count={self.manifest.node_count} differs from {count} updates"
                )
            if (
                count
                and not connection.execute(
                    "SELECT 1 FROM updates WHERE node_id=?", (self.root_node_id,)
                ).fetchone()
            ):
                raise ValueError(f"Generation root {self.root_node_id!r} has no update")
            if self._source_identity() != self.identity:
                raise ValueError("Training export changed during index construction")
            connection.execute("INSERT INTO completed VALUES (?)", (self.identity,))
            connection.commit()
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Training reader index validation failed")
            self._measure_build_disk(temporary)
            connection.close()
            temporary.replace(self.index_path)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Duplicate node ID or update without immutable record: {exc}"
            ) from exc
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)
            with suppress(FileNotFoundError):
                temporary.with_name(temporary.name + "-journal").unlink()

    def _measure_build_disk(self, temporary: Path) -> None:
        """Sample main/journal bytes at transaction boundaries, without walking disk."""
        total = temporary.stat().st_size
        journal = temporary.with_name(temporary.name + "-journal")
        if journal.exists():
            total += journal.stat().st_size
        self.index_build_peak_disk_bytes = max(self.index_build_peak_disk_bytes, total)

    def validate_source_identity(self) -> None:
        """Reject changed exports before publishing a multi-pass consumer's output."""
        if self._source_identity() != self.identity:
            raise ValueError("Training export changed during Dataset extraction")

    @contextmanager
    def temporary_workspace(self) -> Iterator[sqlite3.Connection]:
        """Lend this index connection for disposable disk-backed Dataset tables.

        Only TEMP tables are writable: the completed index stays read-only.
        Callers must drop their tables on exit. No second source index is built.
        """
        self._connection.execute(f"PRAGMA temp.cache_size=-{SQLITE_CACHE_KIB}")
        try:
            yield self._connection
        finally:
            self._connection.rollback()

    def node_count(self) -> int:
        """Return generation update count, excluding unused historical records."""
        return self._count

    def select_sorted_node_window(
        self, start_cursor: str | None, max_nodes: int
    ) -> WindowSelection:
        """Match legacy inclusive lexicographic circular cursor selection exactly."""
        if max_nodes <= 0 or not self._count:
            return (), None, False
        first = str(
            self._connection.execute(
                "SELECT node_id FROM updates ORDER BY node_id LIMIT 1"
            ).fetchone()[0]
        )
        if (
            start_cursor is None
            or not self._connection.execute(
                "SELECT 1 FROM updates WHERE node_id=?", (start_cursor,)
            ).fetchone()
        ):
            start_cursor = first
        limit = min(max_nodes, self._count)
        tail = tuple(
            str(row[0])
            for row in self._connection.execute(
                "SELECT node_id FROM updates WHERE node_id>=? ORDER BY node_id LIMIT ?",
                (start_cursor, limit + 1),
            )
        )
        if len(tail) > limit:
            return tail[:limit], tail[limit], False
        remaining = limit - len(tail)
        head = tuple(
            str(row[0])
            for row in self._connection.execute(
                "SELECT node_id FROM updates ORDER BY node_id LIMIT ?",
                (remaining + 1,),
            )
        )
        return (
            tail + head[:remaining],
            first if max_nodes >= self._count else head[remaining],
            True,
        )

    def load_nodes(self, node_ids: Sequence[str]) -> tuple[TrainingNodeSnapshot, ...]:
        """Reconstruct only requested nodes, in caller order, with the legacy merge."""
        return tuple(self._load_node(node_id) for node_id in node_ids)

    def _load_node(self, node_id: str) -> TrainingNodeSnapshot:
        """Join/decode one immutable payload and one generation update."""
        row = self._connection.execute(
            "SELECT records.payload, updates.payload FROM updates JOIN records USING(node_id) WHERE updates.node_id=?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return _merge_node_record_and_update(
            _node_record_from_dict(json.loads(row[0])),
            _node_update_from_dict(json.loads(row[1])),
        )

    def iter_nodes_in_export_order(self) -> Iterator[TrainingNodeSnapshot]:
        """Yield one node at a time in stable legacy order_index order."""
        for row in self._connection.execute(
            "SELECT node_id FROM updates ORDER BY order_index, sequence"
        ):
            yield self._load_node(str(row[0]))

    def close(self) -> None:
        """Release the read-only index connection."""
        self._connection.close()

    def __enter__(self) -> ShardedTrainingGenerationView:
        """Return this generation view."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the view on success and failure."""
        self.close()
