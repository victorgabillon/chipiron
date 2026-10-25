"""Add allocation observations to the existing serial publication measurement.

All graph generation, publication, reader, Dataset and recovery operations remain
in durable_closed_snapshot.measure. This driver uses its existing cProfile,
StageMeasurement and process counters; it adds the existing inode-aware disk
sampler around each measured operation. Run each invocation in a fresh process.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Event, Thread
from typing import Any
from unittest.mock import patch

from docs.research.atomic_activation.measure import disk
from docs.research.durable_closed_snapshot import measure as existing

_original = existing.measured


def measured(name: str, root: Path, operation: Any) -> Any:
    """Observe actual temporary allocations without changing production behavior."""
    stop = Event()
    peak = {
        "workspace_bytes": 0,
        "private_bytes": 0,
        "sqlite_bytes": 0,
        "unlinked_bytes": 0,
        "samples": 0,
    }
    connections = []
    sqlite_paths: set[str] = set()
    original_connect = sqlite3.connect

    def connect(database: Any, *args: Any, **kwargs: Any) -> Any:
        connections.append(str(database))
        return original_connect(database, *args, **kwargs)

    def sample() -> None:
        while not stop.is_set():
            try:
                all_files = [p for p in root.rglob("*") if p.is_file()]
                groups: dict[str, dict[tuple[int, int], int]] = {
                    key: {}
                    for key in (
                        "workspace_bytes",
                        "private_bytes",
                        "sqlite_bytes",
                        "unlinked_bytes",
                    )
                }
                for path in all_files:
                    stat = path.stat()
                    inode = stat.st_dev, stat.st_ino
                    size = stat.st_blocks * 512
                    groups["workspace_bytes"][inode] = size
                    if "staging" in path.parts or any(
                        part.startswith((".closed-", ".validation-"))
                        for part in path.parts
                    ):
                        groups["private_bytes"][inode] = size
                    if "sqlite" in path.name:
                        groups["sqlite_bytes"][inode] = size
                        sqlite_paths.add(str(path))
                for path in Path("/proc/self/fd").iterdir():
                    stat = path.stat()
                    if stat.st_nlink == 0:
                        groups["unlinked_bytes"][(stat.st_dev, stat.st_ino)] = (
                            stat.st_blocks * 512
                        )
                for key, values in groups.items():
                    peak[key] = max(peak[key], sum(values.values()))
                peak["samples"] += 1
            except FileNotFoundError:
                pass  # Concurrent immutable rename/SQLite journal cleanup.
            stop.wait(0.02)

    def observed() -> Any:
        thread = Thread(target=sample)
        thread.start()
        try:
            with patch.object(sqlite3, "connect", connect):
                return operation()
        finally:
            stop.set()
            thread.join()

    value, observation = _original(name, root, observed)
    observation["allocation_sample_20ms_lower_bound"] = peak
    observation["sqlite_connection_paths"] = connections
    observation["sqlite_sampled_files"] = sorted(sqlite_paths)
    observation["retained_reader_index"] = disk(root / "reader-cache")
    return value, observation


if __name__ == "__main__":
    existing.measured = measured
    existing.main()
