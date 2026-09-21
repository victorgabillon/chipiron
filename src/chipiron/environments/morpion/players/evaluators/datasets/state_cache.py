"""Bounded-memory cache of canonical states from existing value rows.

Only one state and its original target are stored per source row. D4 variants
are rebuilt on demand. SQLite transactions retain completed prefixes after an
interruption, without modifying the source dataset or creating search labels.
"""
# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sqlite3
import time
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.learning import decode_morpion_state_ref_payload
from chipiron.environments.morpion.types import MorpionState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def file_sha256(path: Path) -> str:
    """Return the full-file identity used to reject stale cache reuse."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_state_cache(
    source: Path,
    destination: Path,
    *,
    max_rows: int,
    row_indices: tuple[int, ...] | None = None,
    progress: Callable[[str], None] = print,
) -> CanonicalStateCache:
    """Decode a JSONL prefix once; resume only with identical source/config."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    if row_indices is not None and (
        not row_indices
        or tuple(sorted(set(row_indices))) != row_indices
        or row_indices[0] < 0
        or row_indices[-1] >= max_rows
    ):
        raise ValueError("Selected source indices must be sorted, unique and in range.")
    selected = (
        None
        if row_indices is None
        else {index: position for position, index in enumerate(row_indices)}
    )
    expected_count = max_rows if row_indices is None else len(row_indices)
    identity = json.dumps(
        {
            "schema": "morpion_canonical_state_cache_v1",
            "source": str(source.resolve()),
            "sha256": file_sha256(source),
            "max_rows": max_rows,
            "row_indices": row_indices,
        },
        sort_keys=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY, state BLOB NOT NULL, target REAL NOT NULL)"
        )
        previous = connection.execute("SELECT identity FROM metadata").fetchone()
        if previous is None:
            connection.execute("INSERT INTO metadata VALUES (?)", (identity,))
        elif previous[0] != identity:
            raise ValueError("Canonical cache source/configuration differs.")
        connection.commit()
        completed = int(connection.execute("SELECT COUNT(*) FROM states").fetchone()[0])
        if completed == expected_count:
            return CanonicalStateCache(destination, completed)
        started = last_log = time.monotonic()
        progress(f"Decoding canonical states: {completed}/{expected_count} cached")
        cached = completed
        seen = 0
        with source.open() as stream:
            header = json.loads(next(stream))
            if header.get("kind") != "morpion_supervised_rows_metadata":
                raise ValueError("Expected existing Morpion JSONL metadata.")
            for line in stream:
                if not line.strip():
                    continue
                if seen == max_rows:
                    break
                record = json.loads(line)
                if record.get("kind") != "morpion_supervised_row":
                    raise ValueError("Unexpected Morpion row record.")
                index = seen
                seen += 1
                if selected is not None and index not in selected:
                    continue
                cache_index = index if selected is None else selected[index]
                if cache_index < completed:
                    continue
                row = record["row"]
                target = float(row["target_value"])
                if not math.isfinite(target):
                    raise ValueError("Non-finite value target.")
                state = MorpionState.from_atomheart_state(
                    decode_morpion_state_ref_payload(row["state_ref_payload"]),
                    is_terminal=bool(row["is_terminal"]),
                )
                connection.execute(
                    "INSERT INTO states VALUES (?, ?, ?)",
                    (cache_index, pickle.dumps(state, protocol=5), target),
                )
                cached += 1
                now = time.monotonic()
                if cached % 100 == 0 or now - last_log >= 15:
                    connection.commit()
                if now - last_log >= 15:
                    rate = (cached - completed) / (now - started)
                    progress(
                        f"Decoding {cached}/{expected_count}; {rate:.1f} rows/s; ETA {(expected_count - cached) / rate / 60:.1f} min"
                    )
                    last_log = now
        if seen != max_rows:
            raise ValueError("Dataset contains fewer rows than requested.")
    progress(f"Canonical cache ready: {expected_count} original rows")
    return CanonicalStateCache(destination, expected_count)


class CanonicalStateCache:
    """Read canonical states with a small per-worker SQLite page cache."""

    def __init__(self, path: Path, count: int) -> None:
        """Store immutable cache location; connections open inside each worker."""
        self.path = path
        self.count = count
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        """Return the source prefix length."""
        return self.count

    def __getitem__(self, index: int) -> tuple[MorpionState, float]:
        """Read one trusted locally generated canonical state and target."""
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._connection is None:
            self._connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True
            )
            self._connection.execute("PRAGMA cache_size=-2048")
        result = self._connection.execute(
            "SELECT state, target FROM states WHERE id=?", (index,)
        ).fetchone()
        if result is None:
            raise ValueError("Incomplete canonical state cache.")
        return cast("MorpionState", pickle.loads(result[0])), float(result[1])

    def __getstate__(self) -> dict[str, object]:
        """Never transfer an open database connection to spawned workers."""
        return {"path": self.path, "count": self.count, "_connection": None}
