"""Read published sharded training exports with bounded RAM and disposable SQLite.

Only individual JSON rows are built. V2 canonicality checks decode one state at
a time; indexing never constructs a tree snapshot. Memory is O(window + largest
row/state + parser buffer + SQLite cache + bounded generation metadata); disk
is O(all immutable rows + generation updates + graph validation tables).
"""
# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager, suppress
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ijson
import zstandard
from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.training_export import TrainingNodeSnapshot
from platformdirs import user_cache_path

from .generation_publication_io import (
    assert_artifact_writable,
    generation_read_pin,
    pinned_file_hashes,
)
from .performance import StageMeasurement
from .pipeline_memory import log_pipeline_memory
from .scientific_stream import ENCODING, open_scientific_stream
from .scientific_validation_evidence import canonical_record_is_validated
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
from .sharded_training_v2 import (
    _CODEC,
    REFERENCE_CONTRACT,
    _envelope,
    _integer,
    _path,
    _validate_update,
)
from .sharded_training_v2 import (
    CONTRACT as SCIENTIFIC_CONTRACT,
)
from .sharded_training_v2 import (
    KIND as SCIENTIFIC_KIND,
)
from .sharded_training_v2 import (
    VERSION as SCIENTIFIC_VERSION,
)
from .sharded_training_v2_contract import CLOSED_KIND, CLOSED_LAYOUT, closed_envelope

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    from .sharded_training_export import MorpionShardedTrainingGenerationManifest

LOGGER = logging.getLogger(__name__)
INDEX_SCHEMA_VERSION = 3
PARSER_BUFFER_BYTES = 64 * 1024
SQLITE_CACHE_KIB = 2048
TRANSACTION_ROWS = 256
MANIFEST_MAX_BYTES = 4 * 1024 * 1024

type WindowSelection = tuple[tuple[str, ...], str | None, bool]


def _has_surrogate_escape(path: Path, descriptor: dict[str, Any] | None = None) -> bool:
    """Conservatively choose a lossless library backend, without parsing JSON.

    YAJL replaces unpaired high-surrogate escapes with '?'. Detect all possible
    surrogate escapes, including escaped backslashes, across fixed read chunks.
    False positives only select a slower backend; they never change values.
    """
    found = False
    previous = b""
    with open_scientific_stream(path, descriptor) as stream:
        while chunk := stream.read(PARSER_BUFFER_BYTES):
            data = previous + chunk
            found = found or re.search(rb"\\u[dD][89a-fA-F]", data) is not None
            previous = data[-5:]
    return found


def _parser_events(
    path: Path,
    *,
    multiple_values: bool = False,
    strict_keys: bool = False,
    data: bytes | None = None,
    descriptor: dict[str, Any] | None = None,
) -> Iterator[tuple[str, str, object]]:
    """Share the existing lossless bounded ijson backends across both schemas."""
    lossless_unicode = (
        _has_surrogate_escape(path, descriptor)
        if data is None
        else re.search(rb"\\u[dD][89a-fA-F]", data) is not None
    ) or ijson.backend == "python"
    with (
        open_scientific_stream(path, descriptor)
        if data is None
        else io.BytesIO(data) as stream
    ):
        backend = ijson.get_backend("python") if lossless_unicode else ijson
        events = backend.parse(
            stream,
            buf_size=1 if lossless_unicode else PARSER_BUFFER_BYTES,
            multiple_values=multiple_values,
        )
        maps: list[set[str] | None] = []
        for prefix, event, value in events:
            if strict_keys:
                if event == "start_map":
                    # This one map is O(N); uniqueness is validated on disk.
                    maps.append(None if prefix == "creation_index" else set())
                elif event == "map_key" and maps[-1] is not None:
                    keys = maps[-1]
                    if str(value) in keys:
                        raise ValueError(
                            f"Duplicate JSON key in {path}: {prefix}.{value}"
                        )
                    keys.add(str(value))
                elif event == "end_map":
                    maps.pop()
            yield prefix, event, float(value) if isinstance(value, Decimal) else value


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
    keys: set[str] = set()
    found_array = False
    with _closing_events(_parser_events(path)) as events:
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


@contextmanager
def _closing_events(
    events: Iterator[tuple[str, str, object]],
) -> Iterator[Iterator[tuple[str, str, object]]]:
    """Close file-backed parser generators even when header sniffing stops early."""
    try:
        yield events
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


def iter_shard_rows(
    path: Path,
    field: str,
    *,
    scientific_generation: int | None = None,
    closed: bool = False,
    descriptor: dict[str, Any] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield validated JSON array records, retaining no complete input array."""
    try:
        if scientific_generation is not None:
            with _closing_events(
                _parser_events(
                    path, multiple_values=True, strict_keys=True, descriptor=descriptor
                )
            ) as events:
                rows = ijson.items(events, "")
                header = next(rows, None)
                if (
                    not isinstance(header, dict)
                    or header
                    != (closed_envelope if closed else _envelope)(
                        field, generation=scientific_generation
                    )
                    or not _integer(header.get("format_version"))
                    or not _integer(header.get("generation"))
                ):
                    raise ValueError(f"Unsupported scientific shard envelope: {path}")
                if closed and not _integer(header.get("storage_layout_version")):
                    raise ValueError(f"Unsupported scientific layout version: {path}")
                for row in rows:
                    if not isinstance(row, dict):
                        raise TypeError(f"Expected scientific row object: {path}")
                    yield cast("dict[str, object]", row)
            return
        for row in ijson.items(_shard_events(path, field), f"{field}.item"):
            if not isinstance(row, dict):
                raise TypeError(f"Expected object in {path} {field!r} array")
            yield cast("dict[str, object]", row)
    except ijson.JSONError as exc:
        raise ValueError(f"Malformed or incomplete JSON shard {path}: {exc}") from exc


def _source_kind(path: Path) -> object:
    """Recognize an explicit kind without ever building a creation-index map."""
    kind: object = MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND
    scientific_fields = False
    with _closing_events(_parser_events(path)) as events:
        for prefix, event, value in events:
            if prefix == "format_kind" and event not in ("map_key", "end_map"):
                kind = value
                if kind in (SCIENTIFIC_KIND, CLOSED_KIND):
                    return kind
            if prefix in ("scientific_contract", "creation_index"):
                scientific_fields = True
    if scientific_fields:
        raise ValueError("Missing or ambiguous explicit scientific format kind")
    return kind


def _scientific_manifest(
    path: Path, role: str, *, closed: bool = False
) -> dict[str, Any]:
    """Bound metadata, streaming past v2's O(N) creation_index without loading it."""

    def metadata_events() -> Iterator[tuple[str, str, object]]:
        skipping = False
        started = False
        size = 0
        for prefix, event, value in _parser_events(path, strict_keys=True):
            if prefix == "" and event == "map_key" and value == "creation_index":
                skipping, started = True, False
                yield prefix, event, value
                yield "creation_index", "null", None
                continue
            if skipping:
                if not started:
                    if (prefix, event) != ("creation_index", "start_map"):
                        raise ValueError("Scientific creation_index must be an object")
                    started = True
                elif (prefix, event) == ("creation_index", "end_map"):
                    skipping = False
                continue
            size += len(str(value).encode("utf-8", errors="surrogatepass")) + 16
            if size > MANIFEST_MAX_BYTES:
                raise ValueError(
                    "Scientific metadata exceeds bounded-reader size limit"
                )
            yield prefix, event, value

    try:
        values = ijson.items(metadata_events(), "")
        data = next(values)
        if next(values, None) is not None:
            raise ValueError("Multiple scientific manifest objects")
    except (ijson.JSONError, StopIteration) as exc:
        raise ValueError(f"Malformed scientific manifest: {path}") from exc
    if not isinstance(data, dict) or any(
        data.get(key) != value
        for key, value in (closed_envelope if closed else _envelope)(role).items()
    ):
        raise ValueError(f"Unsupported scientific format/contract/role: {path}")
    if not _integer(data["format_version"]):
        raise ValueError("Scientific version must be an integer")
    return cast("dict[str, Any]", data)


def read_closed_manifest(path: Path) -> dict[str, Any]:
    """Strict bounded header for layout 1, independent of the historical catalog."""
    from .sharded_training_v2 import _producer, _require

    manifest = _scientific_manifest(path, "generation", closed=True)
    _require(
        set(manifest)
        == set(closed_envelope("generation"))
        | {"generation", "snapshot", "producer", "record_shard", "update_shard"},
        "missing or unknown closed snapshot field",
    )
    _require(_integer(manifest["storage_layout_version"]), "invalid layout version")
    generation = manifest["generation"]
    _require(_integer(generation) and generation >= 1, "invalid generation")
    producer = manifest["producer"]
    _require(
        isinstance(producer, dict) and isinstance(producer.get("identity"), dict),
        "invalid producer",
    )
    _require(
        producer == _producer(producer["identity"]),
        "incompatible producer/codec identity",
    )
    info = manifest["snapshot"]
    _require(
        isinstance(info, dict)
        and set(info)
        == {"root_node_id", "node_count", "created_at_unix_s", "metadata"},
        "invalid snapshot header",
    )
    _require(
        isinstance(info["root_node_id"], str) and bool(info["root_node_id"]),
        "invalid root",
    )
    _require(
        _integer(info["node_count"]) and info["node_count"] > 0, "invalid node count"
    )
    _require(isinstance(info["metadata"], dict), "invalid snapshot metadata")
    _require(
        info["created_at_unix_s"] is None
        or (
            type(info["created_at_unix_s"]) in (int, float)
            and math.isfinite(info["created_at_unix_s"])
        ),
        "invalid snapshot timestamp",
    )
    _require(
        info["metadata"].get("format_kind") == "training_tree_snapshot"
        and _integer(info["metadata"].get("format_version"))
        and info["metadata"]["format_version"] == 1
        and info["metadata"].get("default_target_source")
        in ("direct_value", "tree_value", "effective_value"),
        "invalid snapshot metadata contract",
    )
    for key, name in (
        ("record_shard", "records.jsonl"),
        ("update_shard", "updates.jsonl"),
    ):
        item = manifest[key]
        compressed = isinstance(item, dict) and "encoding" in item
        if compressed:
            _require(item["encoding"] == ENCODING, "unsupported scientific encoding")
            name += ".zst"
            _require(
                all(
                    _integer(item.get(k)) and item[k] > 0
                    for k in ("bytes", "decoded_bytes")
                )
                and isinstance(item.get("decoded_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["decoded_sha256"]) is not None,
                "invalid compressed content identity",
            )
        _require(
            isinstance(item, dict)
            and set(item)
            == {"path", "generation", "count", "sha256"}
            | (
                {"encoding", "bytes", "decoded_bytes", "decoded_sha256"}
                if compressed
                else set()
            ),
            "invalid closed shard descriptor",
        )
        _require(
            item["path"] == name
            and _integer(item["generation"])
            and item["generation"] == generation,
            "invalid closed shard path/generation",
        )
        _require(
            _integer(item["count"]) and item["count"] == info["node_count"],
            "invalid closed shard count",
        )
        _require(
            isinstance(item["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
            "invalid closed shard hash",
        )
        _require(
            not (path.parent / name).is_symlink(),
            "symlinked closed snapshot dependency",
        )
    _require(
        manifest["record_shard"].get("encoding")
        == manifest["update_shard"].get("encoding"),
        "mixed scientific encodings",
    )
    return manifest


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
    """Recognize explicit schemas and keep flat snapshots on the legacy path."""
    kind = _source_kind(path)
    if kind == SCIENTIFIC_KIND:
        _scientific_manifest(path, "generation")
        return True
    if kind != MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND:
        raise ValueError(f"Unsupported explicit training-export format: {kind}")
    if path.stat().st_size > MANIFEST_MAX_BYTES:
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and all(
        key in payload for key in ("node_shard_path", "update_shard_path", "generation")
    )


def is_scientific_generation_manifest(path: Path) -> bool:
    """Identify v2 for explicit routing refusal, without building its manifest."""
    return _source_kind(path) == SCIENTIFIC_KIND


def require_scientific_opt_in(allowed: bool) -> None:
    """Refuse legacy fallback for explicitly requested experimental v2 routing."""
    if not allowed:
        raise ValueError(
            "Scientific v2 requires explicit opt-in and a supported bounded consumer; production certainty gate remains unresolved"
        )


def _configure_connection(connection: sqlite3.Connection) -> None:
    """Bound page cache and sorting RAM; never mmap the whole index."""
    connection.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KIB}")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA temp.cache_size=-{SQLITE_CACHE_KIB}")
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
        allow_scientific_v2: bool = False,
        allow_closed_snapshot: bool = False,
    ) -> None:
        """Initialize/reuse an index without constructing training snapshots."""
        self.index_build_peak_disk_bytes = 0
        # Format preparation runs inside the pin's exception-safe initializer.
        # Declare the fields populated by those format-specific helpers here.
        self.index_build_temp_logical_bytes: int
        self.format_version: int
        self.scientific_contract: str | None
        self.created_at_unix_s: float | None
        self.provenance: dict[str, Any]
        self.snapshot_metadata: dict[str, Any]
        self._v2_manifest: dict[str, Any]
        self._v2_descriptors: dict[Path, dict[str, Any]]
        self.storage_layout_version: int | None
        self.scientific_encoding: str | None
        self.root: Path
        self.manifest: MorpionShardedTrainingGenerationManifest
        self.root_node_id: str
        self.generation: int
        self._expected_count: int
        self._node_shards: list[Path]
        self._source_paths: list[Path]
        self._update_shard: Path
        self._publication_pin = generation_read_pin(Path(generation_manifest_path))
        self._publication_pin.__enter__()
        try:
            self._publication_hashes = pinned_file_hashes(
                Path(generation_manifest_path)
            )
            self._initialize(
                generation_manifest_path,
                cache_dir=cache_dir,
                allow_scientific_v2=allow_scientific_v2,
                allow_closed_snapshot=allow_closed_snapshot,
            )
        except BaseException:
            self._publication_pin.__exit__(None, None, None)
            raise

    def _initialize(
        self,
        generation_manifest_path: str | Path,
        *,
        cache_dir: str | Path | None,
        allow_scientific_v2: bool,
        allow_closed_snapshot: bool,
    ) -> None:
        """Build the selected format's view after acquiring its lifetime pin."""
        measurement = StageMeasurement()
        self.source_bytes_read = 0
        self.index_build_temp_logical_bytes = 0
        self.manifest_path = Path(generation_manifest_path).resolve()
        self.format_kind = _source_kind(self.manifest_path)
        self.format_version = 1
        self.storage_layout_version = None
        self.scientific_encoding = None
        self.scientific_contract = None
        self.created_at_unix_s = None
        self.provenance = {}
        self.snapshot_metadata = {
            "format_kind": "training_tree_snapshot",
            "format_version": 1,
        }
        self._v2_manifest = {}
        self._v2_descriptors = {}
        self._v2_prepared_stats: dict[Path, tuple[int, int, int]] = {}
        self._payload_decoder = zstandard.ZstdDecompressor()
        self.root = self.manifest_path.parent
        if self.format_kind == CLOSED_KIND:
            if not allow_closed_snapshot:
                raise ValueError(
                    "Closed snapshots require explicit allow_closed_snapshot"
                )
            stat = self.manifest_path.stat()
            self._v2_prepared_stats[self.manifest_path] = (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            self._prepare_closed_source()
        elif self.format_kind == SCIENTIFIC_KIND:
            if not allow_scientific_v2:
                raise ValueError(
                    "Scientific v2 requires explicit allow_scientific_v2; certainty activation gate remains unresolved"
                )
            for path in (
                self.manifest_path,
                self.manifest_path.parent.parent / "manifest.json",
            ):
                stat = path.stat()
                self._v2_prepared_stats[path] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            self._prepare_scientific_source()
        else:
            self._prepare_legacy_source()
        self.identity = self._source_identity()
        directory = (
            Path(cache_dir)
            if cache_dir is not None
            else user_cache_path("chipiron") / "sharded_training_reader"
        ).resolve()
        # Even an explicit override must never place a cache inside the export.
        if directory == self.root or self.root in directory.parents:
            raise ValueError("Reader cache must be outside the source export directory")
        if self.format_kind in (SCIENTIFIC_KIND, CLOSED_KIND):
            assert_artifact_writable(directory)
        from chipiron.utils.path_runtime import assert_test_output_path

        assert_test_output_path(directory)
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
            index_build_temp_logical_bytes=self.index_build_temp_logical_bytes,
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

    def _prepare_legacy_source(self) -> None:
        """Retain the v1 manifest selection and interpretation without repair."""
        _read_manifest_identity(self.manifest_path)
        self.manifest = _load_generation_manifest(self.manifest_path)
        self.source_bytes_read += 2 * self.manifest_path.stat().st_size
        if self.manifest.generation < 1:
            raise ValueError("Sharded export generation must be positive")
        self.root_node_id = self.manifest.root_node_id
        self.generation = self.manifest.generation
        self._expected_count = self.manifest.node_count
        self._node_shards = []
        self._source_paths = [self.manifest_path]
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
        self.provenance = {
            "source_format_kind": self.format_kind,
            "source_format_version": 1,
            "source_generation": self.generation,
        }

    def _prepare_scientific_source(self) -> None:
        """Discover explicit v2 source files; never materialize creation_index."""
        manifest = _scientific_manifest(self.manifest_path, "generation")
        expected = set(_envelope("generation")) | {
            "generation",
            "snapshot",
            "producer",
            "creation_index",
            "record_shards",
            "update_shard",
        }
        if set(manifest) != expected:
            raise ValueError(
                "Missing, unknown or ambiguous scientific generation fields"
            )
        self.root = self.manifest_path.parent.parent
        catalog_path = self.root / "manifest.json"
        catalog = _scientific_manifest(catalog_path, "catalog")
        if set(catalog) != set(_envelope("catalog")) | {
            "latest",
            "v2_generations",
            "producer",
        }:
            raise ValueError("Missing, unknown or ambiguous scientific catalog fields")
        self.generation = manifest["generation"]
        if not _integer(self.generation) or self.generation < 1:
            raise ValueError("Invalid scientific generation")
        if (
            catalog["v2_generations"].get(str(self.generation))
            != self.manifest_path.relative_to(self.root).as_posix()
        ):
            raise ValueError("Unpublished scientific generation")
        if manifest["producer"] != catalog["producer"]:
            raise ValueError("Mixed scientific producer identity")
        self._set_scientific_header(manifest)
        self._v2_manifest = manifest
        self._node_shards = []
        self._source_paths = [self.manifest_path, catalog_path]
        for descriptor in manifest["record_shards"]:
            path = self._scientific_descriptor(descriptor)
            if (
                descriptor["generation"] > self.generation
                or path in self._v2_descriptors
            ):
                raise ValueError("Future or duplicate scientific record shard")
            self._node_shards.append(path)
            self._v2_descriptors[path] = descriptor
        self._update_shard = self._scientific_descriptor(manifest["update_shard"])
        if (
            manifest["update_shard"]["generation"] != self.generation
            or self._update_shard in self._v2_descriptors
        ):
            raise ValueError("Mixed scientific update generation/path")
        self._v2_descriptors[self._update_shard] = manifest["update_shard"]
        self._source_paths.extend([*self._node_shards, self._update_shard])
        self.source_bytes_read += 2 * (
            self.manifest_path.stat().st_size + catalog_path.stat().st_size
        )

    def _set_scientific_header(self, manifest: dict[str, Any]) -> None:
        """Share value semantics across the historical and closed layouts."""
        if (
            importlib.metadata.version("atomheart") != "0.2.11"
            or not isinstance(manifest["producer"], dict)
            or manifest["producer"].get("state_ref_contract") != REFERENCE_CONTRACT
        ):
            raise ValueError("Unsupported scientific state-reference contract/decoder")
        info = manifest["snapshot"]
        if set(info) != {"root_node_id", "node_count", "created_at_unix_s", "metadata"}:
            raise ValueError("Incomplete scientific snapshot header")
        if not _integer(info["node_count"]) or info["node_count"] < 1:
            raise ValueError("Invalid scientific node count")
        self.root_node_id = _validated_node_id(info["root_node_id"])
        self._expected_count = info["node_count"]
        self.snapshot_metadata = dict(info["metadata"])
        self.created_at_unix_s = (
            None
            if info["created_at_unix_s"] is None
            else float(info["created_at_unix_s"])
        )
        self.format_version = SCIENTIFIC_VERSION
        self.scientific_contract = SCIENTIFIC_CONTRACT
        self.provenance = {
            "source_format_kind": SCIENTIFIC_KIND,
            "source_format_version": SCIENTIFIC_VERSION,
            "scientific_contract": SCIENTIFIC_CONTRACT,
            "source_generation": self.generation,
            "producer": manifest["producer"],
        }

    def _prepare_closed_source(self) -> None:
        """Read one self-contained descriptor; there is no historical catalog."""
        manifest = read_closed_manifest(self.manifest_path)
        self.scientific_encoding = manifest["record_shard"].get("encoding")
        self.root = self.manifest_path.parent
        self.generation = manifest["generation"]
        self._set_scientific_header(manifest)
        self.storage_layout_version = CLOSED_LAYOUT
        self.provenance.update(
            source_format_kind=CLOSED_KIND, storage_layout_version=CLOSED_LAYOUT
        )
        self._v2_manifest = manifest
        self._node_shards = [self.root / manifest["record_shard"]["path"]]
        self._update_shard = self.root / manifest["update_shard"]["path"]
        self._v2_descriptors = {
            self._node_shards[0]: manifest["record_shard"],
            self._update_shard: manifest["update_shard"],
        }
        self._source_paths = [self.manifest_path, *self._v2_descriptors]

    def _scientific_descriptor(self, descriptor: dict[str, Any]) -> Path:
        if (
            set(descriptor) != {"path", "generation", "count", "sha256"}
            or not _integer(descriptor["generation"])
            or not _integer(descriptor["count"])
            or descriptor["count"] < 0
            or descriptor["generation"] < 1
        ):
            raise ValueError("Invalid scientific shard descriptor")
        return _path(self.root, descriptor["path"])

    def _source_identity(self) -> str:
        """Hash exact source contents with a fixed-size buffer on every open."""
        sources: list[dict[str, object]] = []
        for path in self._source_paths:
            before = path.stat()
            prepared = self._v2_prepared_stats.get(path)
            expected = self._publication_hashes.get(str(path))
            if prepared is not None and (
                prepared[:2] != (before.st_size, before.st_mtime_ns)
                or (expected is None and prepared[2] != before.st_ctime_ns)
            ):
                raise ValueError(
                    f"Training export changed while preparing scientific metadata: {path}"
                )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.source_bytes_read += before.st_size
            after = path.stat()
            expected = self._publication_hashes.get(str(path))
            if expected is not None and digest != expected:
                raise ValueError(f"Committed training export changed: {path}")
            changed_payload = (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            )
            # Linking immutable payloads into a new capsule changes inode ctime.
            # Exact committed hashes replace that mutable-file heuristic here.
            changed_legacy_metadata = (
                expected is None and before.st_ctime_ns != after.st_ctime_ns
            )
            if changed_payload or changed_legacy_metadata:
                raise ValueError(
                    f"Training export changed during fingerprinting: {path}"
                )
            if (
                self.format_version == SCIENTIFIC_VERSION
                and path in self._v2_descriptors
                and digest != self._v2_descriptors[path]["sha256"]
            ):
                raise ValueError(f"Scientific source shard hash mismatch: {path}")
            sources.append({
                "path": str(path.resolve()),
                "size": before.st_size,
                "sha256": digest,
            })
        return json.dumps(
            {
                "schema": INDEX_SCHEMA_VERSION,
                "format": self.format_version,
                "format_kind": self.format_kind,
                "scientific_contract": self.scientific_contract,
                "generation": self.generation,
                "producer": self.provenance.get("producer"),
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
                valid = bool(
                    row == (self.identity,)
                    # quick_check misses secondary indexes inconsistent with rows.
                    and connection.execute("PRAGMA integrity_check").fetchone()
                    == ("ok",)
                )
                if valid and self.format_version == SCIENTIFIC_VERSION:
                    valid = connection.execute(
                        "SELECT digest FROM cache_digest"
                    ).fetchone() == (self._cache_digest(connection),)
                return valid
            finally:
                connection.close()
        except (sqlite3.DatabaseError, TypeError, ValueError):
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
                CREATE TABLE cache_digest (digest TEXT NOT NULL);
            """)
            count = 0
            if self.format_version == SCIENTIFIC_VERSION:
                self._build_scientific_rows(connection, temporary)
            for path in (
                () if self.format_version == SCIENTIFIC_VERSION else self._node_shards
            ):
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
                ()
                if self.format_version == SCIENTIFIC_VERSION
                else iter_shard_rows(self._update_shard, "updates")
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
            if count != self._expected_count:
                raise ValueError(
                    f"Generation node_count={self._expected_count} differs from {count} updates"
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
            if self.format_version == SCIENTIFIC_VERSION:
                connection.execute(
                    "INSERT INTO cache_digest VALUES (?)",
                    (self._cache_digest(connection),),
                )
            connection.execute("INSERT INTO completed VALUES (?)", (self.identity,))
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
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

    @staticmethod
    def _cache_digest(connection: sqlite3.Connection) -> str:
        """Verify compressed bytes before decoding any cached scientific payload."""
        digest = hashlib.sha256()
        for table, fields in (
            ("records", "node_id,payload"),
            ("updates", "node_id,order_index,sequence,payload"),
        ):
            digest.update(table.encode())
            for row in connection.execute(
                f"SELECT {fields} FROM {table} ORDER BY node_id"
            ):
                payload = row[-1]
                if not isinstance(payload, bytes):
                    raise TypeError("Scientific cache payload is not a BLOB")
                digest.update(json.dumps(row[:-1], ensure_ascii=True).encode("utf-8"))
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
        return digest.hexdigest()

    def _build_scientific_rows(
        self, connection: sqlite3.Connection, temporary: Path
    ) -> None:
        """Use the same index and parser, with disk-backed v2 contract checks."""
        compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)

        def pack(row: dict[str, Any]) -> bytes:
            return compressor.compress(json.dumps(row, allow_nan=False).encode())

        connection.executescript("""
            CREATE TEMP TABLE v2_creation (id TEXT PRIMARY KEY, generation INTEGER NOT NULL);
            CREATE TEMP TABLE v2_parents (parent TEXT, child TEXT, PRIMARY KEY(parent,child));
            CREATE TEMP TABLE v2_children (parent TEXT, child TEXT, PRIMARY KEY(parent,child));
        """)
        count = 0
        for path in self._node_shards:
            descriptor = self._v2_descriptors[path]
            shard_count = 0
            canonical_validated = canonical_record_is_validated(
                self.manifest_path, self.provenance["producer"], descriptor
            )
            for row in iter_shard_rows(
                path,
                "records",
                scientific_generation=descriptor["generation"],
                closed=self.storage_layout_version is not None,
                descriptor=self._v2_descriptors[path],
            ):
                if set(row) != {
                    "node_id",
                    "creation_generation",
                    "state_ref_payload",
                } or not isinstance(row["node_id"], str):
                    raise ValueError("Invalid scientific immutable record")
                if not _integer(row["creation_generation"]) or (
                    not 1 <= cast("int", row["creation_generation"]) <= self.generation
                    if self.storage_layout_version is not None
                    else row["creation_generation"] != descriptor["generation"]
                ):
                    raise ValueError("Invalid scientific creation generation")
                ref = row["state_ref_payload"]
                if not canonical_validated:
                    canonical = checkpoint_payload_to_jsonable(
                        _CODEC.dump_state_ref(_CODEC.load_state_ref(ref))
                    )
                    if ref is None or canonical != ref:
                        raise ValueError("Noncanonical scientific state reference")
                connection.execute(
                    "INSERT INTO records VALUES (?, ?)",
                    (
                        _validated_node_id(row["node_id"]),
                        pack(row),
                    ),
                )
                count += 1
                shard_count += 1
                if count % TRANSACTION_ROWS == 0:
                    self._measure_build_disk(temporary)
                    connection.commit()
            if shard_count != descriptor["count"]:
                raise ValueError("Scientific record shard count mismatch")
            self.source_bytes_read += 2 * path.stat().st_size
        if self.storage_layout_version is not None:
            if count != self._expected_count:
                raise ValueError("Closed snapshot record count mismatch")
        else:
            with _closing_events(
                _parser_events(self.manifest_path, strict_keys=True)
            ) as events:
                for index, (node_id, generation) in enumerate(
                    ijson.kvitems(events, "creation_index"), 1
                ):
                    if (
                        not _integer(generation)
                        or not 1 <= generation <= self.generation
                    ):
                        raise ValueError("Invalid scientific creation index entry")
                    connection.execute(
                        "INSERT INTO v2_creation VALUES (?, ?)",
                        (_validated_node_id(node_id), generation),
                    )
                    if index % TRANSACTION_ROWS == 0:
                        connection.commit()
            self.source_bytes_read += 2 * self.manifest_path.stat().st_size
            invalid = any(
                generation is None
                or self._decode_payload(payload)["creation_generation"] != generation
                for payload, generation in connection.execute("""
                    SELECT records.payload, v2_creation.generation FROM records
                    LEFT JOIN v2_creation ON records.node_id=v2_creation.id
                """)
            )
            if (
                invalid
                or count
                != connection.execute("SELECT COUNT(*) FROM v2_creation").fetchone()[0]
            ):
                raise ValueError("Incomplete or inconsistent scientific creation index")
        update_count = 0
        for sequence, row in enumerate(
            iter_shard_rows(
                self._update_shard,
                "updates",
                scientific_generation=self.generation,
                closed=self.storage_layout_version is not None,
                descriptor=self._v2_descriptors[self._update_shard],
            )
        ):
            _validate_update(row)
            if row["order_index"] != sequence:
                raise ValueError("Invalid scientific export order")
            node_id = _validated_node_id(cast("str", row["node_id"]))
            connection.execute(
                "INSERT INTO updates VALUES (?, ?, ?, ?)",
                (node_id, sequence, sequence, pack(row)),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO v2_parents VALUES (?, ?)",
                (
                    (_validated_node_id(parent), node_id)
                    for parent in cast("list[str]", row["parent_ids"])
                ),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO v2_children VALUES (?, ?)",
                (
                    (node_id, _validated_node_id(child))
                    for child in cast("list[str]", row["child_ids"])
                ),
            )
            update_count += 1
            if update_count % TRANSACTION_ROWS == 0:
                self._measure_build_disk(temporary)
                connection.commit()
        if update_count != self._v2_descriptors[self._update_shard]["count"]:
            raise ValueError("Scientific update shard count mismatch")
        self._validate_scientific_topology(connection, update_count)
        self.index_build_temp_logical_bytes = int(
            connection.execute("PRAGMA temp.page_count").fetchone()[0]
        ) * int(connection.execute("PRAGMA temp.page_size").fetchone()[0])
        connection.executescript("""
            DROP TABLE v2_creation; DROP TABLE v2_parents; DROP TABLE v2_children;
            DROP TABLE v2_degrees; DROP TABLE v2_queue;
        """)

    @staticmethod
    def _validate_scientific_topology(
        connection: sqlite3.Connection, count: int
    ) -> None:
        """Require the reference reader's reciprocal closed DAG using TEMP disk."""
        for left, right in (
            ("v2_parents", "v2_children"),
            ("v2_children", "v2_parents"),
        ):
            if connection.execute(
                f"SELECT * FROM {left} EXCEPT SELECT * FROM {right} LIMIT 1"
            ).fetchone():
                raise ValueError("Nonreciprocal scientific topology")
        if connection.execute("""
            SELECT 1 FROM v2_parents e LEFT JOIN updates p ON p.node_id=e.parent
            LEFT JOIN updates c ON c.node_id=e.child
            WHERE p.node_id IS NULL OR c.node_id IS NULL LIMIT 1
        """).fetchone():
            raise ValueError("Scientific topology references nodes outside generation")
        connection.executescript("""
            CREATE TEMP TABLE v2_degrees (id TEXT PRIMARY KEY, degree INTEGER);
            INSERT INTO v2_degrees SELECT u.node_id, COUNT(p.parent) FROM updates u
                LEFT JOIN v2_parents p ON p.child=u.node_id GROUP BY u.node_id;
            CREATE TEMP TABLE v2_queue (id TEXT PRIMARY KEY);
            INSERT INTO v2_queue SELECT id FROM v2_degrees WHERE degree=0;
        """)
        visited = 0
        while row := connection.execute(
            "SELECT id FROM v2_queue ORDER BY id LIMIT 1"
        ).fetchone():
            connection.execute("DELETE FROM v2_queue WHERE id=?", row)
            connection.execute(
                "UPDATE v2_degrees SET degree=degree-1 WHERE id IN (SELECT child FROM v2_children WHERE parent=?)",
                row,
            )
            connection.execute(
                "INSERT OR IGNORE INTO v2_queue SELECT id FROM v2_degrees JOIN v2_children ON id=child WHERE parent=? AND degree=0",
                row,
            )
            visited += 1
            if visited % TRANSACTION_ROWS == 0:
                connection.commit()
        if visited != count:
            raise ValueError("Cyclic scientific topology")

    @staticmethod
    def _scientific_node(
        record: dict[str, Any], update: dict[str, Any]
    ) -> TrainingNodeSnapshot:
        """Reconstruct exactly the saved v2 fields, without a v1 value fallback."""
        # Ingestion validates every update before the completion marker. Opening
        # verifies source identity and all cached bytes; published indexes are
        # immutable. Repeating that field validation on every Dataset pass adds
        # no evidence within this validated view's lifetime.
        return TrainingNodeSnapshot(
            node_id=update["node_id"],
            parent_ids=tuple(update["parent_ids"]),
            child_ids=tuple(update["child_ids"]),
            depth=update["depth"],
            state_ref_payload=record["state_ref_payload"],
            direct_value_scalar=None
            if update["direct_value_scalar"] is None
            else float(update["direct_value_scalar"]),
            tree_value_scalar=None
            if update["tree_value_scalar"] is None
            else float(update["tree_value_scalar"]),
            effective_value_scalar=None
            if update["effective_value_scalar"] is None
            else float(update["effective_value_scalar"]),
            effective_value_source=update["effective_value_source"],
            target_value_scalar=None
            if update["target_value_scalar"] is None
            else float(update["target_value_scalar"]),
            backed_up_value_scalar=None
            if update["backed_up_value_scalar"] is None
            else float(update["backed_up_value_scalar"]),
            is_terminal=update["is_terminal"],
            is_exact=update["is_exact"],
            over_event_label=update["over_event_label"],
            visit_count=update["visit_count"],
            metadata=dict(update["metadata"]),
        )

    def load_value_contracts(
        self, node_ids: Sequence[str]
    ) -> tuple[dict[str, Any], ...]:
        """Expose complete certainty/target contracts for a caller-bounded window."""
        if self.format_version != SCIENTIFIC_VERSION:
            raise ValueError("Per-value scientific contracts are unavailable in v1")
        result = []
        for node_id in node_ids:
            row = self._connection.execute(
                "SELECT payload FROM updates WHERE node_id=?", (node_id,)
            ).fetchone()
            if row is None:
                raise KeyError(node_id)
            result.append(self._decode_payload(row[0])["value_contract"])
        return tuple(result)

    def load_creation_generations(self, node_ids: Sequence[str]) -> tuple[int, ...]:
        """Return original scientific creation identities for a bounded window."""
        if self.format_version != SCIENTIFIC_VERSION:
            raise ValueError("Scientific creation origins require v2")
        result = []
        for node_id in node_ids:
            row = self._connection.execute(
                "SELECT payload FROM records WHERE node_id=?", (node_id,)
            ).fetchone()
            if row is None:
                raise KeyError(node_id)
            result.append(self._decode_payload(row[0])["creation_generation"])
        return tuple(result)

    def iter_record_bytes(self) -> Iterator[tuple[str, bytes]]:
        """Stream exact JSONL record bytes under this view's verified identity.

        Offline copying requires one object per line, as emitted by both writers.
        Historical multiline JSON remains readable through the ordinary view.
        The existing strict lossless parser validates each frame; no new parser.
        """
        for path in self._node_shards:
            descriptor = self._v2_descriptors[path]
            envelope = (
                closed_envelope
                if self.storage_layout_version is not None
                else _envelope
            )
            # Exhaustion or generator.close() exits this context, including on copy failure.
            with open_scientific_stream(path, descriptor) as stream:  # pylint: disable=contextmanager-generator-missing-cleanup
                for index, raw in enumerate(stream):
                    with _closing_events(
                        _parser_events(path, strict_keys=True, data=raw)
                    ) as events:
                        rows = ijson.items(events, "")
                        row = next(rows, None)
                        if (
                            not isinstance(row, dict)
                            or next(rows, None) is not None
                            or not raw.endswith(b"\n")
                        ):
                            raise ValueError(
                                "Copy source requires complete single-line JSONL records"
                            )
                    if index == 0:
                        if row != envelope(
                            "records", generation=descriptor["generation"]
                        ):
                            raise ValueError("Copy source header changed")
                    else:
                        yield row["node_id"], raw
        self.validate_source_identity()

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
            message = (
                "Training export changed during Dataset extraction"
                if self.format_version == 1
                else "Training export changed during scientific consumption"
            )
            raise ValueError(message)

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
        if self.format_version == SCIENTIFIC_VERSION:
            return self._scientific_node(
                self._decode_payload(row[0]), self._decode_payload(row[1])
            )
        return _merge_node_record_and_update(
            _node_record_from_dict(json.loads(row[0])),
            _node_update_from_dict(json.loads(row[1])),
        )

    def _decode_payload(self, payload: bytes) -> dict[str, Any]:
        """Decode one scientific row; never decompress a complete index."""
        return cast(
            "dict[str, Any]",
            json.loads(
                self._payload_decoder.decompress(payload, allow_extra_data=False)
            ),
        )

    def iter_nodes_in_export_order(self) -> Iterator[TrainingNodeSnapshot]:
        """Yield one node at a time in stable legacy order_index order."""
        for row in self._connection.execute(
            "SELECT node_id FROM updates ORDER BY order_index, sequence"
        ):
            yield self._load_node(str(row[0]))

    def close(self) -> None:
        """Release the read-only index connection."""
        try:
            self._connection.close()
        finally:
            self._publication_pin.__exit__(None, None, None)

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
