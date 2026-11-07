"""Owned, bounded SQLite storage for one derived evaluator-input generation.

No scientific authority. The lease spans every reader and replacement. Only
complete, verified entries survive publication; dead staging is disposable.
"""

# Exact types deliberately exclude bool/int coercion and altered subclasses.
# pylint: disable=unidiomatic-typecheck

import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import psutil

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    assert_artifact_writable,
    file_digest,
    publish_directory_if_absent,
    require,
    sync_directory,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    HANDCRAFTED_20_CORE_FEATURE_NAMES,
    MORPION_CANONICAL_FEATURE_NAMES,
)

FORMAT = "morpion_evaluator_inputs_v1"
MAX_PAYLOAD = 128 * 1024
METADATA_BYTES = 64 * 1024


def encoded(value: object) -> bytes:
    """Deterministic finite metadata only; no state deserialization."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write(path: Path, value: object) -> None:
    data = encoded(value)
    require(len(data) <= METADATA_BYTES, "Oversized cache metadata")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path: Path) -> Any:
    require(path.stat().st_size <= METADATA_BYTES, "Oversized cache metadata")
    return json.loads(path.read_bytes())


def _regular(path: Path) -> None:
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_uid == os.getuid(),
        "Unsafe evaluator cache file",
    )


def safe_layout(root: Path) -> None:
    """Refuse all symlink components/children before reading or removing storage."""
    require(
        root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked evaluator cache path",
    )
    require(
        root.is_dir()
        and root.stat().st_uid == os.getuid()
        and not root.stat().st_mode & 0o022,
        "Unsafe evaluator cache owner",
    )
    for path in root.iterdir():
        require(not path.is_symlink(), "Symlink in evaluator cache")
        if path.name in {"entry", "building"}:
            require(
                path.is_dir() and path.stat().st_uid == os.getuid(),
                "Unsafe evaluator cache entry",
            )
            for child in path.iterdir():
                _regular(child)
                require(
                    child.name in {"inputs.sqlite", "complete.json"},
                    "Foreign evaluator cache entry child",
                )
        else:
            require(path.name in {"OWNER", ".lock"}, "Foreign evaluator cache root")
            _regular(path)
    require(
        _json(root / "OWNER") == {"format": FORMAT}, "Foreign evaluator cache owner"
    )


@contextmanager
def ownership(root: Path) -> Iterator[bool]:
    """Stable nonblocking lease; contention makes no data/pruning mutations."""
    import fcntl

    require(
        root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked evaluator cache path",
    )
    assert_artifact_writable(root)
    # An existing unmarked directory is foreign, even if empty. Incomplete root
    # creation can be removed manually; never adopt unrelated operator storage.
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    else:
        _write(root / "OWNER", {"format": FORMAT})
        sync_directory(root)
        sync_directory(root.parent)
    _regular(root / "OWNER")
    require(
        _json(root / "OWNER") == {"format": FORMAT}, "Foreign evaluator cache owner"
    )
    require(
        root.stat().st_uid == os.getuid() and not root.stat().st_mode & 0o022,
        "Unsafe evaluator cache owner",
    )
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        _regular(root / ".lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        identity = (root.stat().st_dev, root.stat().st_ino)
        safe_layout(root)
        if (root / "building").exists():
            shutil.rmtree(root / "building")
            sync_directory(root)
        try:
            yield True
        finally:
            safe_layout(root)
            require(
                (root.stat().st_dev, root.stat().st_ino) == identity
                and (root / ".lock").stat().st_ino == os.fstat(fd).st_ino,
                "Evaluator cache ownership changed",
            )
    finally:
        os.close(fd)


def resources(root: Path, limit: int) -> bool:
    """Reserve full prior/replacement capacity, metadata and resident headroom."""
    return (
        shutil.disk_usage(root.parent).free
        >= 2 * limit + 3 * METADATA_BYTES + 20 * 2**30
        and psutil.Process().memory_info().rss + 32 * 2**20 <= 4 * 2**30
        and psutil.virtual_memory().available >= 64 * 2**20
    )


def _connect(path: Path, *, readonly: bool, limit: int) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True
    )
    try:
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_PAYLOAD + 4096)
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA trusted_schema=OFF")
        if readonly:
            connection.execute("PRAGMA query_only=ON")
        else:
            # Only private disposable staging: no rollback/WAL/journal coexistence.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(f"PRAGMA max_page_count={limit // 4096}")
            connection.execute(
                "CREATE TABLE inputs (key TEXT PRIMARY KEY, payload BLOB NOT NULL, raw BLOB NOT NULL, binding TEXT NOT NULL)"
            )
    except BaseException:
        connection.close()
        raise
    return connection


def binding(recipe: bytes, payload: bytes, raw: bytes) -> str:
    """Bind every raw FP32 row to its exact opaque state and input recipe."""
    return hashlib.sha256(recipe + payload + raw).hexdigest()


def _input_width(identity: dict[str, object]) -> int:
    """Bind raw shape to one canonical ordered representation, never infer it."""
    name, shape = identity.get("representation"), identity.get("shape")
    width = 20 if name == "handcrafted_20_core" else 41
    features = (
        HANDCRAFTED_20_CORE_FEATURE_NAMES
        if width == 20
        else MORPION_CANONICAL_FEATURE_NAMES
    )
    require(
        identity.get("format") == FORMAT
        and name in ("handcrafted_20_core", "handcrafted_41")
        and type(shape) is list
        and len(shape) == 1
        and type(shape[0]) is int
        and shape == [width]
        and identity.get("features") == list(features)
        and identity.get("dtype") == "float32"
        and identity.get("device") == "cpu"
        and identity.get("layout") == "contiguous",
        "Unsupported evaluator input representation",
    )
    return width


def valid_row(row: tuple[Any, ...], recipe: bytes, width: int) -> None:
    """Validate every exact opaque-state/raw-FP32 binding before use."""
    key, payload, raw, digest = row
    require(
        type(payload) is bytes
        and len(payload) <= MAX_PAYLOAD
        and type(raw) is bytes
        and len(raw) == width * 4
        and hashlib.sha256(payload).hexdigest() == key
        and binding(recipe, payload, raw) == digest
        and all(math.isfinite(x) for x in struct.unpack(f"={width}f", raw)),
        "Invalid evaluator input binding",
    )


def verify(
    entry: Path, identity: dict[str, object], limit: int, max_rows: int
) -> dict[str, Any]:
    """Bounded complete-file, schema and every-row verification before reuse."""
    width = _input_width(identity)
    require(
        {p.name for p in entry.iterdir()} == {"inputs.sqlite", "complete.json"},
        "Incomplete evaluator inputs",
    )
    manifest = _json(entry / "complete.json")
    require(
        type(manifest) is dict
        and set(manifest) == {"identity", "rows", "bytes", "sha256"}
        and manifest["identity"] == identity
        and type(manifest["rows"]) is int
        and 0 <= manifest["rows"] <= max_rows
        and type(manifest["bytes"]) is int
        and (entry / "inputs.sqlite").stat().st_size == manifest["bytes"] <= limit,
        "Incompatible evaluator input completion",
    )
    require(
        file_digest(entry / "inputs.sqlite") == manifest["sha256"],
        "Evaluator input checksum mismatch",
    )
    connection = _connect(entry / "inputs.sqlite", readonly=True, limit=limit)
    try:
        require(
            set(connection.execute("SELECT type,name FROM sqlite_schema"))
            == {("table", "inputs"), ("index", "sqlite_autoindex_inputs_1")},
            "Invalid evaluator input schema",
        )
        require(
            connection.execute("SELECT count(*) FROM inputs").fetchone()
            == (manifest["rows"],),
            "Invalid evaluator input count",
        )
        recipe = encoded(identity)
        for row in connection.execute("SELECT key,payload,raw,binding FROM inputs"):
            valid_row(row, recipe, width)
    finally:
        connection.close()
    return cast("dict[str, Any]", manifest)


class InputStore:
    """One leased prior and bounded private replacement, never an authority."""

    def __init__(
        self,
        root: Path,
        identity: dict[str, object],
        limit: int,
        max_rows: int,
        observation: dict[str, object],
    ) -> None:
        self.root, self.identity, self.limit, self.max_rows = (
            root,
            identity,
            limit,
            max_rows,
        )
        self.recipe = encoded(identity)
        self.input_width = _input_width(identity)
        self.observation = observation
        self.prior: sqlite3.Connection | None = None
        self.connection: sqlite3.Connection | None = None
        self.manifest: dict[str, Any] | None = None
        self.manifest_digest: str | None = None
        self.requests = 0
        self.entry, self.stage = root / "entry", root / "building"
        try:
            if self.entry.exists():
                try:
                    self.manifest = verify(self.entry, identity, limit, max_rows)
                except ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError:
                    observation["prior"] = "incompatible-or-corrupt"
                else:
                    self.prior = _connect(
                        self.entry / "inputs.sqlite", readonly=True, limit=limit
                    )
                    self.manifest_digest = file_digest(self.entry / "complete.json")
                    observation["prior"] = "verified"
            else:
                observation["prior"] = "missing"
            # Retain any prior until successful replacement; size admission includes it.
            prior_bytes = (
                sum(p.stat().st_size for p in self.entry.iterdir())
                if self.entry.exists()
                else 0
            )
            require(
                prior_bytes <= limit + METADATA_BYTES,
                "Prior evaluator cache exceeds budget",
            )
            observation["prior_bytes"] = prior_bytes
            self.stage.mkdir()
            self.connection = _connect(
                self.stage / "inputs.sqlite", readonly=False, limit=limit
            )
        except BaseException:
            self.close()
            raise

    def lookup(self, payload: bytes) -> bytes | None:
        """Verify the exact payload even when its indexed hash agrees."""
        self.requests += 1
        require(
            self.requests <= self.max_rows, "Evaluator cache request budget exhausted"
        )
        if self.prior is None:
            return None
        key = hashlib.sha256(payload).hexdigest()
        row = self.prior.execute(
            "SELECT key,payload,raw,binding FROM inputs WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        valid_row(row, self.recipe, self.input_width)
        require(row[1] == payload, "Evaluator cache exact payload mismatch")
        return cast("bytes", row[2])

    def append(self, payload: bytes, raw: bytes) -> None:
        """Bound actual SQLite pages; no truncation or fallback after consumption."""
        assert self.connection is not None
        self.connection.execute(
            "INSERT OR IGNORE INTO inputs VALUES (?,?,?,?)",
            (
                hashlib.sha256(payload).hexdigest(),
                payload,
                raw,
                binding(self.recipe, payload, raw),
            ),
        )
        if self.requests % 128 == 0:
            require(
                resources(self.root, self.limit),
                "Evaluator cache resource budget exhausted",
            )
        require(
            (self.stage / "inputs.sqlite").stat().st_size <= self.limit,
            "Evaluator cache disk budget exhausted",
        )

    def finish(self) -> None:
        """Verify after consumption, close readers, then atomically complete storage."""
        safe_layout(self.root)
        if self.manifest is not None:
            require(
                file_digest(self.entry / "complete.json") == self.manifest_digest
                and verify(self.entry, self.identity, self.limit, self.max_rows)
                == self.manifest,
                "Evaluator prior changed during consumption",
            )
        assert self.connection is not None
        self.connection.commit()
        rows = self.connection.execute("SELECT count(*) FROM inputs").fetchone()[0]
        self.connection.close()
        self.connection = None
        path = self.stage / "inputs.sqlite"
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _write(
            self.stage / "complete.json",
            {
                "identity": self.identity,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            },
        )
        verify(self.stage, self.identity, self.limit, self.max_rows)
        self.close()
        if self.entry.exists():
            shutil.rmtree(self.entry)
            sync_directory(self.root)
        sync_directory(self.stage)
        publish_directory_if_absent(self.stage, self.entry, self.root)
        self.observation["entry_bytes"] = sum(
            p.stat().st_size for p in self.entry.iterdir()
        )
        self.observation["unique_rows"] = rows

    def close(self) -> None:
        """End all SQLite consumers; failed staging is recovered under the lease."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.prior is not None:
            self.prior.close()
            self.prior = None
