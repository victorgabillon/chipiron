"""Deterministic identity checks for the validated checkpoint payload cache."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from anemone.checkpoints import SearchRuntimeCheckpointPayload, TreeCheckpointPayload

from chipiron.environments.morpion.bootstrap.runtime import checkpoint_io

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("change", ["path", "size", "mtime", "missing"])
def test_validated_payload_cache_rejects_changed_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """A real file identity change invalidates and releases a previously valid entry."""
    monkeypatch.setattr(
        checkpoint_io,
        "_validated_checkpoint_payload_cache",
        checkpoint_io._ValidatedCheckpointPayloadCache(),
    )
    path = tmp_path / "checkpoint.json"
    path.write_text("{}", encoding="utf-8")
    payload = SearchRuntimeCheckpointPayload(
        evaluator_version=1, tree=TreeCheckpointPayload(root_node_id=0)
    )
    checkpoint_io.cache_morpion_search_checkpoint_payload_for_restore(
        str(path), payload
    )
    assert checkpoint_io._pop_cached_morpion_search_checkpoint_payload_for_restore(
        path
    ) == (payload, 2)
    assert (
        checkpoint_io._pop_cached_morpion_search_checkpoint_payload_for_restore(path)
        is None
    )
    checkpoint_io.cache_morpion_search_checkpoint_payload_for_restore(path, payload)

    if change == "path":
        path = tmp_path / "different.json"
        path.write_text("{}", encoding="utf-8")
    elif change == "size":
        path.write_text("{}\n", encoding="utf-8")
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        path.unlink()

    assert (
        checkpoint_io._pop_cached_morpion_search_checkpoint_payload_for_restore(path)
        is None
    )
    assert checkpoint_io._validated_checkpoint_payload_cache.entry is None
