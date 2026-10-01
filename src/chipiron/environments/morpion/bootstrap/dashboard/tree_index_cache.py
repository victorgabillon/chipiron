"""Persistent read-only cache for dashboard runtime-checkpoint inspection."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

from anemone.checkpoints import (
    AlgorithmNodeCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    checkpoint_payload_to_jsonable,
    deserialize_checkpoint_atom,
)
from dacite import Config, from_dict
from platformdirs import user_cache_path

from chipiron.environments.morpion.bootstrap.runtime.checkpoint_codec import (
    _normalize_algorithm_node_payload_for_dacite,
)

from .checkpoint_reader import read_inspection_checkpoint

LOGGER = logging.getLogger(__name__)

_INDEX_SCHEMA_VERSION = 1
_NODE_CACHE_LIMIT = 512


class PersistentTreeIndexCacheError(ValueError):
    """Raised when the derived dashboard tree index cannot be used."""


class IndexedChildLink:
    """One indexed branch edge from a checkpoint node to an expanded child."""

    __slots__ = ("branch_key", "child_node_id")

    def __init__(self, *, branch_key: object, child_node_id: int) -> None:
        self.branch_key = branch_key
        self.child_node_id = child_node_id


class PersistentCheckpointTreeIndex:
    """Random-access view over one derived SQLite checkpoint index."""

    __slots__ = (
        "root_node_id",
        "database_path",
        "_node_cache",
        "_parent_cache",
        "_child_link_cache",
    )

    def __init__(self, *, root_node_id: int, database_path: Path) -> None:
        self.root_node_id = root_node_id
        self.database_path = database_path
        self._node_cache: dict[int, AlgorithmNodeCheckpointPayload] = {}
        self._parent_cache: dict[int, tuple[int, ...]] = {}
        self._child_link_cache: dict[int, tuple[IndexedChildLink, ...]] = {}

    def has_node(self, node_id: int) -> bool:
        """Return whether one node id exists without scanning the checkpoint."""
        if node_id in self._node_cache:
            return True
        try:
            with _connect(self.database_path) as connection:
                row = connection.execute(
                    "SELECT 1 FROM nodes WHERE node_id = ? LIMIT 1", (node_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise PersistentTreeIndexCacheError(str(exc)) from exc
        return row is not None

    def node(self, node_id: int) -> AlgorithmNodeCheckpointPayload:
        """Load one typed node payload by id."""
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached
        try:
            with _connect(self.database_path) as connection:
                row = connection.execute(
                    "SELECT payload_json FROM nodes WHERE node_id = ?", (node_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise PersistentTreeIndexCacheError(str(exc)) from exc
        if row is None:
            raise KeyError(node_id)
        node_payload = _deserialize_node_payload(str(row[0]))
        _remember(self._node_cache, node_id, node_payload)
        return node_payload

    def node_or_none(self, node_id: int) -> AlgorithmNodeCheckpointPayload | None:
        """Load one node by id when present."""
        try:
            return self.node(node_id)
        except KeyError:
            return None

    def parent_ids(self, node_id: int) -> tuple[int, ...]:
        """Return reverse parent ids for one node."""
        cached = self._parent_cache.get(node_id)
        if cached is not None:
            return cached
        try:
            with _connect(self.database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT parent_node_id
                    FROM parents
                    WHERE child_node_id = ?
                    ORDER BY parent_node_id
                    """,
                    (node_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise PersistentTreeIndexCacheError(str(exc)) from exc
        result = tuple(int(row[0]) for row in rows)
        _remember(self._parent_cache, node_id, result)
        return result

    def child_links(self, node_id: int) -> tuple[IndexedChildLink, ...]:
        """Return expanded child links for one node."""
        cached = self._child_link_cache.get(node_id)
        if cached is not None:
            return cached
        node_payload = self.node(node_id)
        result = tuple(
            IndexedChildLink(
                branch_key=deserialize_checkpoint_atom(linked_child.branch_key),
                child_node_id=linked_child.child_node_id,
            )
            for linked_child in node_payload.linked_children
        )
        _remember(self._child_link_cache, node_id, result)
        return result


def persistent_checkpoint_tree_index_exists(checkpoint_path: Path) -> bool:
    """Return whether a valid derived index already exists without building it."""
    checkpoint_path = checkpoint_path.resolve()
    try:
        identity = _checkpoint_identity(checkpoint_path)
    except OSError:
        return False
    target_path = _index_path(checkpoint_path, identity)
    return _load_existing_index(target_path, identity) is not None


def load_or_build_persistent_checkpoint_tree_index(
    checkpoint_path: Path,
) -> PersistentCheckpointTreeIndex:
    """Return a persistent random-access index, building it once per checkpoint."""
    checkpoint_path = checkpoint_path.resolve()
    identity = _checkpoint_identity(checkpoint_path)
    target_path = _index_path(checkpoint_path, identity)
    started_at = time.perf_counter()
    cached = _load_existing_index(target_path, identity)
    if cached is not None:
        LOGGER.info(
            "[dashboard] persistent_tree_index_hit checkpoint=%s elapsed=%.3fs bytes=%s",
            checkpoint_path,
            time.perf_counter() - started_at,
            target_path.stat().st_size,
        )
        return cached

    read_started_at = time.perf_counter()
    payload = read_inspection_checkpoint(checkpoint_path)
    read_elapsed_s = time.perf_counter() - read_started_at
    build_started_at = time.perf_counter()
    try:
        _build_index(target_path, identity=identity, payload=payload)
        build_elapsed_s = time.perf_counter() - build_started_at
        cached = _load_existing_index(target_path, identity)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise PersistentTreeIndexCacheError(
            f"failed to build persistent tree index: {exc}"
        ) from exc
    if cached is None:
        raise PersistentTreeIndexCacheError(
            "persistent tree index did not validate after creation"
        )
    _remove_stale_indexes(target_path)
    LOGGER.info(
        "[dashboard] persistent_tree_index_built checkpoint=%s nodes=%s "
        "checkpoint_read=%.3fs index_build=%.3fs total=%.3fs bytes=%s",
        checkpoint_path,
        len(payload.tree.nodes),
        read_elapsed_s,
        build_elapsed_s,
        time.perf_counter() - started_at,
        target_path.stat().st_size,
    )
    return cached


def _dashboard_cache_root() -> Path:
    """Return the dashboard-only user cache directory."""
    override = os.environ.get("CHIPIRON_DASHBOARD_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return user_cache_path("chipiron") / "bootstrap"


def _checkpoint_identity(checkpoint_path: Path) -> str:
    """Return a cheap immutable identity for one completed checkpoint artifact."""
    identity_path = (
        checkpoint_path / "manifest.json"
        if checkpoint_path.is_dir()
        else checkpoint_path
    )
    stat = identity_path.stat()
    return json.dumps(
        {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "checkpoint_path": str(checkpoint_path),
            "identity_path": str(identity_path),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _index_path(checkpoint_path: Path, identity: str) -> Path:
    """Return a bounded per-run path for one checkpoint index."""
    namespace_source = str(checkpoint_path.parent.resolve()).encode("utf-8")
    namespace = hashlib.sha256(namespace_source).hexdigest()[:20]
    identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (
        _dashboard_cache_root() / "tree-index" / namespace / f"{identity_hash}.sqlite3"
    )


def _connect(database_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite index."""
    if not database_path.is_file():
        raise PersistentTreeIndexCacheError(
            f"persistent tree index is missing: {database_path}"
        )
    return sqlite3.connect(database_path)


def _load_existing_index(
    database_path: Path,
    identity: str,
) -> PersistentCheckpointTreeIndex | None:
    """Validate cheap metadata and return one existing persistent index."""
    if not database_path.is_file():
        return None
    try:
        with sqlite3.connect(database_path) as connection:
            metadata = dict(
                connection.execute("SELECT key, value FROM metadata").fetchall()
            )
            if metadata.get("schema_version") != str(_INDEX_SCHEMA_VERSION):
                return None
            if metadata.get("checkpoint_identity") != identity:
                return None
            root_node_id = int(metadata["root_node_id"])
            root_exists = connection.execute(
                "SELECT 1 FROM nodes WHERE node_id = ? LIMIT 1", (root_node_id,)
            ).fetchone()
            if root_exists is None:
                return None
    except (KeyError, ValueError, sqlite3.Error):
        LOGGER.warning(
            "[dashboard] persistent_tree_index_invalid path=%s",
            database_path,
            exc_info=True,
        )
        try:
            database_path.unlink()
        except OSError:
            pass
        return None
    return PersistentCheckpointTreeIndex(
        root_node_id=root_node_id,
        database_path=database_path,
    )


def _build_index(
    database_path: Path,
    *,
    identity: str,
    payload: SearchRuntimeCheckpointPayload,
) -> None:
    """Build one SQLite index atomically from a fully validated checkpoint."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_name(f".{database_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.unlink()
    except FileNotFoundError:
        pass

    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE parents (
                child_node_id INTEGER NOT NULL,
                parent_node_id INTEGER NOT NULL,
                PRIMARY KEY (child_node_id, parent_node_id)
            );
            CREATE INDEX parents_by_child ON parents(child_node_id);
            """
        )
        connection.executemany(
            "INSERT INTO nodes(node_id, payload_json) VALUES (?, ?)",
            (
                (
                    node.node_id,
                    json.dumps(
                        _inspection_node_mapping(node),
                        separators=(",", ":"),
                    ),
                )
                for node in payload.tree.nodes
            ),
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO parents(child_node_id, parent_node_id)
            VALUES (?, ?)
            """,
            (
                (linked_child.child_node_id, node.node_id)
                for node in payload.tree.nodes
                for linked_child in node.linked_children
            ),
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", str(_INDEX_SCHEMA_VERSION)),
                ("checkpoint_identity", identity),
                ("root_node_id", str(payload.tree.root_node_id)),
                ("node_count", str(len(payload.tree.nodes))),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary_path, database_path)


def _remove_stale_indexes(current_path: Path) -> None:
    """Keep only the current checkpoint index in one runtime-checkpoint namespace."""
    for candidate in current_path.parent.glob("*.sqlite3"):
        if candidate == current_path:
            continue
        try:
            candidate.unlink()
        except OSError:
            LOGGER.debug(
                "[dashboard] persistent_tree_index_prune_failed path=%s",
                candidate,
                exc_info=True,
            )


def _inspection_node_mapping(
    node: AlgorithmNodeCheckpointPayload,
) -> dict[str, object]:
    """Keep only node fields required by dashboard state/value/navigation reads."""
    raw = checkpoint_payload_to_jsonable(node)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint node payload must serialize to an object")
    required_keys = (
        "node_id",
        "parent_node_id",
        "branch_from_parent",
        "depth",
        "state_payload",
        "generated_all_branches",
        "linked_children",
        "evaluation",
    )
    return {key: raw[key] for key in required_keys}


def _deserialize_node_payload(payload_json: str) -> AlgorithmNodeCheckpointPayload:
    """Reconstruct one typed Anemone node payload from the SQLite row."""
    raw_payload = json.loads(payload_json)
    if not isinstance(raw_payload, dict):
        raise PersistentTreeIndexCacheError("node payload must be a JSON object")
    normalized = _normalize_algorithm_node_payload_for_dacite(
        cast("dict[str, Any]", raw_payload)
    )
    return from_dict(
        AlgorithmNodeCheckpointPayload,
        cast("dict[str, Any]", normalized),
        config=Config(cast=[tuple], check_types=False),
    )


def _remember(cache: dict[int, Any], key: int, value: Any) -> None:
    """Insert into one tiny FIFO cache without retaining a whole checkpoint."""
    cache[key] = value
    if len(cache) > _NODE_CACHE_LIMIT:
        oldest_key = next(iter(cache))
        del cache[oldest_key]


__all__ = [
    "IndexedChildLink",
    "PersistentCheckpointTreeIndex",
    "PersistentTreeIndexCacheError",
    "load_or_build_persistent_checkpoint_tree_index",
    "persistent_checkpoint_tree_index_exists",
]
