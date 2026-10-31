"""Bounded operational storage for exact StateSplit results, never authority.

Sixteen fixed slots give each key a stable kernel lock (collisions serialize).
Each slot retains one completed artifact and at most one private build. A reader
loads an immutable value while holding that lock; no returned object references
cache files. Replacement/abandoned-build cleanup therefore cannot evict a live
reader or writer. Lock files are never removed. The namespace is cache-owned;
unknown entries and corrupt completions fail closed rather than being deleted.
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .state_split_cache_identity import FORMAT, digest, encoded, semantic_fingerprint

LOGGER = logging.getLogger(__name__)
SLOTS = 16
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_INDICES = 1_000_000
MAX_MANIFEST_BYTES = 4096
LOCK_TIMEOUT_SECONDS = 600.0
SplitValue = tuple[frozenset[int], int, str]
Builder = Callable[[str, str, float, int, int | None], SplitValue]
_PRIVATE = re.compile(r"(?:attempt|operation)-[0-9a-f]{32}")
_HEX = re.compile(r"[0-9a-f]{64}")
_ENTERED: ContextVar[bool] = ContextVar("split_cache_entered", default=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), "invalid lock file")
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    message = "State split cache lock wait exhausted"
                    raise TimeoutError(message) from None
                time.sleep(0.05)
        yield
    finally:
        os.close(descriptor)


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _plain_directory(path: Path) -> None:
    _require(not path.is_symlink(), "directory symlink is not owned cache storage")
    path.mkdir(exist_ok=True)
    _require(path.is_dir(), "expected owned directory")


def _root(configured: Path, source: Path) -> Path:
    root = configured.expanduser().resolve()
    source = source.resolve()
    _require(
        not root.is_relative_to(source.parent),
        "cache must be outside dataset directory",
    )
    for ancestor in (root, *root.parents):
        _require(
            not ancestor.name.startswith((
                "dataset_publications_v",
                "generation_publications_v",
                "training_publications_v",
            ))
            and not any(ancestor.glob(".generation_publication_v*.json")),
            "cache must be outside immutable publication storage",
        )
    root.mkdir(parents=True, exist_ok=True)
    namespace = root / FORMAT
    _plain_directory(namespace)
    with _lock(namespace / ".namespace.lock"):
        marker = namespace / "OWNER"
        if not marker.exists():
            _require(
                {p.name for p in namespace.iterdir()}
                <= {".namespace.lock", ".owner-pending"},
                "refusing to claim nonempty unmarked storage",
            )
            pending = namespace / ".owner-pending"
            if pending.exists():
                _require(
                    not pending.is_symlink() and pending.is_file(),
                    "invalid owner staging",
                )
                pending.unlink()
            _write(pending, FORMAT.encode())
            os.rename(pending, marker)
            _sync(namespace)
            _sync(root)
        _require(
            not marker.is_symlink() and _read(marker, 64) == FORMAT.encode(),
            "invalid namespace owner",
        )
    return namespace


def _slot(namespace: Path, key: str) -> Path:
    return namespace / f"slot-{int(key[:8], 16) % SLOTS:02d}"


def _cleanup(slot: Path) -> None:
    """Only under the stable slot lock; never inspect or clean other namespaces."""
    for path in slot.iterdir():
        _require(not path.is_symlink(), "symlink in owned slot")
        if path.name in {"entry", "operation.json"}:
            continue
        _require(_PRIVATE.fullmatch(path.name) is not None, "unknown slot entry")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_pairs)
    _require(isinstance(value, dict), "expected JSON object")
    assert isinstance(value, dict)
    return value


def _read(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        _require(
            stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
            "artifact must be a regular file",
        )
        _require(
            os.fstat(stream.fileno()).st_size <= limit, "artifact exceeds size bound"
        )
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, "artifact changed beyond size bound")
    return raw


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _validate_identity(identity: dict[str, Any]) -> None:
    _require(
        set(identity)
        == {"format", "source", "semantics", "policy", "fraction", "seed", "limit"},
        "identity fields",
    )
    _require(
        identity["format"] == FORMAT
        and identity["policy"] == "canonical_state_sha256_v1",
        "incompatible format/policy",
    )
    _require(
        _is_digest(identity["source"]) and _is_digest(identity["semantics"]),
        "invalid content/semantic fingerprint",
    )
    _require(_integer(identity["seed"]), "invalid seed")
    limit = identity["limit"]
    _require(limit is None or (_integer(limit) and limit >= 0), "invalid limit")
    _require(isinstance(identity["fraction"], str), "invalid fraction")
    fraction = float.fromhex(identity["fraction"])
    _require(
        math.isfinite(fraction)
        and 0 <= fraction <= 1
        and fraction.hex() == identity["fraction"],
        "invalid fraction",
    )


def _load(entry: Path) -> tuple[dict[str, Any], SplitValue]:
    _require(not entry.is_symlink() and entry.is_dir(), "invalid completed entry")
    _require(
        {p.name for p in entry.iterdir()} == {"split.json", "complete.json"},
        "incomplete/unknown completed entry",
    )
    manifest = _object(_read(entry / "complete.json", MAX_MANIFEST_BYTES))
    _require(
        set(manifest) == {"identity", "key", "payload_sha256"}, "completion fields"
    )
    identity = manifest["identity"]
    _require(isinstance(identity, dict), "invalid completion identity")
    _validate_identity(identity)
    _require(
        manifest["key"] == hashlib.sha256(encoded(identity)).hexdigest(),
        "identity checksum mismatch",
    )
    raw = _read(entry / "split.json", MAX_PAYLOAD_BYTES)
    _require(
        hashlib.sha256(raw).hexdigest() == manifest["payload_sha256"],
        "payload checksum mismatch",
    )
    value = _object(raw)
    _require(
        set(value)
        == {"indices", "rows", "validation_count", "train_count", "population"},
        "payload fields",
    )
    count, indices, population = value["rows"], value["indices"], value["population"]
    _require(_integer(count) and 0 <= count <= sys.maxsize, "invalid row count")
    _require(
        identity["limit"] is None or count <= identity["limit"],
        "row count exceeds limit",
    )
    _require(
        isinstance(indices, list) and len(indices) <= MAX_INDICES, "invalid indices"
    )
    previous = -1
    for index in indices:
        _require(
            _integer(index) and previous < index < count,
            "unordered/out-of-bounds index",
        )
        previous = index
    _require(
        _integer(value["validation_count"])
        and value["validation_count"] == len(indices),
        "validation count mismatch",
    )
    _require(
        _integer(value["train_count"]) and value["train_count"] == count - len(indices),
        "train count mismatch",
    )
    _require(_is_digest(population), "invalid population digest")
    return identity, (frozenset(indices), count, population)


@dataclass(frozen=True)
class _Result:
    value: SplitValue
    outcome: str
    build_s: float = 0.0
    load_s: float = 0.0
    publish_s: float = 0.0
    verify_s: float = 0.0
    lock_wait_s: float = 0.0


@lru_cache(maxsize=2)
def _cached(
    namespace: Path, path: str, identity_bytes: bytes, builder: Builder
) -> _Result:
    _ENTERED.set(True)
    identity = _object(identity_bytes)
    key = hashlib.sha256(identity_bytes).hexdigest()
    slot = _slot(namespace, key)
    started = time.perf_counter()
    with _lock(namespace / f"{slot.name}.lock"):
        waited = time.perf_counter() - started
        _plain_directory(slot)
        _sync(namespace)
        _cleanup(slot)
        entry = slot / "entry"
        if entry.exists():
            started = time.perf_counter()
            found, value = _load(entry)
            loaded = time.perf_counter() - started
            if found == identity:
                return _Result(value, "hit", load_s=loaded, lock_wait_s=waited)
            # A valid different key collides in this bounded slot. It is never
            # consumed; all prior consumers already own independent values.
            shutil.rmtree(entry)
            _sync(slot)
        attempt = slot / f"attempt-{uuid.uuid4().hex}"
        attempt.mkdir()
        started = time.perf_counter()
        value = builder(
            path,
            identity["source"],
            float.fromhex(identity["fraction"]),
            identity["seed"],
            identity["limit"],
        )
        built = time.perf_counter() - started
        started = time.perf_counter()
        _require(
            digest(Path(path)) == identity["source"], "source changed during build"
        )
        verified = time.perf_counter() - started
        if len(value[0]) > MAX_INDICES:
            attempt.rmdir()
            return _Result(
                value,
                "uncached-size-limit",
                build_s=built,
                verify_s=verified,
                lock_wait_s=waited,
            )
        started = time.perf_counter()
        raw = encoded({
            "indices": sorted(value[0]),
            "rows": value[1],
            "population": value[2],
            "validation_count": len(value[0]),
            "train_count": value[1] - len(value[0]),
        })
        if len(raw) > MAX_PAYLOAD_BYTES:
            attempt.rmdir()
            return _Result(
                value,
                "uncached-size-limit",
                build_s=built,
                verify_s=verified,
                lock_wait_s=waited,
            )
        _write(attempt / "split.json", raw)
        _write(
            attempt / "complete.json",
            encoded({
                "identity": identity,
                "key": key,
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
            }),
        )
        _require(_load(attempt) == (identity, value), "publication verification failed")
        _sync(attempt)
        os.rename(attempt, entry)
        _sync(slot)
        return _Result(
            value,
            "miss",
            build_s=built,
            publish_s=time.perf_counter() - started,
            verify_s=verified,
            lock_wait_s=waited,
        )


def load_or_build(
    configured: Path,
    source: Path,
    fraction: float,
    seed: int,
    limit: int | None,
    policy: str,
    builder: Builder,
) -> SplitValue:
    """Return exact immutable membership and bounded operational-only telemetry.

    Source pins remain the caller's existing responsibility through consumption.
    Hash both sides even for a memory hit. A resource-sized result remains an exact
    uncached result; no row limit or altered population is introduced by this cache.
    """
    started = time.perf_counter()
    namespace = _root(configured, source)
    tick = time.perf_counter()
    semantics = semantic_fingerprint()
    semantics_s = time.perf_counter() - tick
    tick = time.perf_counter()
    source_digest = digest(source)
    hash_s = time.perf_counter() - tick
    identity = {
        "format": FORMAT,
        "source": source_digest,
        "semantics": semantics,
        "policy": policy,
        "fraction": float(fraction).hex(),
        "seed": seed,
        "limit": limit,
    }
    _validate_identity(identity)
    identity_bytes = encoded(identity)
    key = hashlib.sha256(identity_bytes).hexdigest()
    token = _ENTERED.set(False)
    try:
        result = _cached(namespace, str(source), identity_bytes, builder)
        memory_hit = not _ENTERED.get()
    finally:
        _ENTERED.reset(token)
    tick = time.perf_counter()
    _require(digest(source) == source_digest, "source changed during consumption")
    verify_s = time.perf_counter() - tick
    observation = {
        "key": key,
        "source_path": str(source),
        "outcome": "memory-hit" if memory_hit else result.outcome,
        "semantic_verification_s": semantics_s,
        "source_hash_s": hash_s,
        "source_verify_s": verify_s + (0.0 if memory_hit else result.verify_s),
        "build_s": 0.0 if memory_hit else result.build_s,
        "load_s": 0.0 if memory_hit else result.load_s,
        "publish_s": 0.0 if memory_hit else result.publish_s,
        "lock_wait_s": 0.0 if memory_hit else result.lock_wait_s,
        "elapsed_before_record_s": time.perf_counter() - started,
    }
    slot = _slot(namespace, key)
    with _lock(namespace / f"{slot.name}.lock"):
        _plain_directory(slot)
        _cleanup(slot)
        pending = slot / f"operation-{uuid.uuid4().hex}"
        _write(pending, encoded(observation))
        os.replace(pending, slot / "operation.json")
        _sync(slot)
    LOGGER.info("state_split_cache %s", encoded(observation).decode())
    return result.value
