"""Compact disposable caches retain scientific integrity and process lifetimes."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import zstandard

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    capsule_lease,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    publish_closed_patch,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.test_morpion_closed_scientific_snapshot import closed


def _open(source: Path, cache: Path) -> ShardedTrainingGenerationView:
    return ShardedTrainingGenerationView(
        source, cache_dir=cache, allow_closed_snapshot=True
    )


def test_scientific_validation_precedes_reuse_and_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ingested update is validated once; a pinned immutable view reuses it."""
    from chipiron.environments.morpion.bootstrap import (
        sharded_training_reader as reader,
    )

    source = closed(tmp_path / "source", encoding="jsonl-zstd-1")
    oracle = load_closed_scientific_snapshot(source)
    validated = []
    original = reader._validate_update

    def checked(row: Any) -> None:
        original(row)
        validated.append(row["node_id"])

    monkeypatch.setattr(reader, "_validate_update", checked)
    with _open(source, tmp_path / "cache") as initial:
        assert validated == [node.node_id for node in oracle.snapshot.nodes]
        for _ in range(3):
            assert tuple(initial.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert len(validated) == len(oracle.snapshot.nodes)
    validated.clear()
    with _open(source, tmp_path / "cache") as reused:
        assert reused.index_reused
        assert tuple(reused.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert not validated
        index = reused.index_path
    # Corruption cannot inherit the previously validated view's lifetime.
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE updates SET sequence=sequence+1")
    with _open(source, tmp_path / "cache") as rebuilt:
        assert not rebuilt.index_reused
        assert validated == [node.node_id for node in oracle.snapshot.nodes]
        assert tuple(rebuilt.iter_nodes_in_export_order()) == oracle.snapshot.nodes


@pytest.mark.parametrize(
    "damage",
    (
        "text",
        "frame",
        "oversized_frame",
        "order",
        "btree",
        "completion",
        "digest",
        "missing",
    ),
)
def test_corrupted_compact_cache_rebuilt_before_payload_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Even valid SQLite and forged frame size cannot bypass the content digest."""
    source = closed(tmp_path / "source", encoding="jsonl-zstd-1")
    oracle = load_closed_scientific_snapshot(source)
    cache = tmp_path / "cache"
    with _open(source, cache) as initial:
        index = initial.index_path
        assert tuple(initial.iter_nodes_in_export_order()) == oracle.snapshot.nodes
    with sqlite3.connect(index) as connection:
        assert connection.execute(
            "SELECT DISTINCT typeof(payload) FROM records"
        ).fetchall() == [("blob",)]
        payload = connection.execute("SELECT payload FROM records LIMIT 1").fetchone()[
            0
        ]
        assert zstandard.ZstdDecompressor().decompress(payload).startswith(b"{")
        if damage in ("text", "frame", "oversized_frame"):
            corrupted = {
                "text": "{}",
                "frame": payload[:-1] + bytes([payload[-1] ^ 1]),
                # A standard header advertising a huge size: never fed to libzstd.
                "oversized_frame": b"\x28\xb5\x2f\xfd\xe0" + b"\xff" * 8,
            }[damage]
            connection.execute("UPDATE records SET payload=?", (corrupted,))
        elif damage == "order":
            connection.execute("UPDATE updates SET sequence=sequence+1")
        elif damage == "btree":
            # Model inconsistent secondary keys without changing table/digest bytes.
            connection.execute("DROP INDEX export_order")
            connection.execute(
                "CREATE INDEX export_order ON updates(-order_index, sequence)"
            )
            connection.execute("PRAGMA writable_schema=ON")
            connection.execute(
                "UPDATE sqlite_schema SET sql="
                "'CREATE INDEX export_order ON updates(order_index, sequence)' "
                "WHERE name='export_order'"
            )
        elif damage == "completion":
            connection.execute("DELETE FROM completed")
        elif damage == "digest":
            connection.execute("UPDATE cache_digest SET digest='invalid'")
    if damage == "missing":
        index.unlink()
    elif damage == "btree":
        with sqlite3.connect(index) as connection:
            assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
            assert connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
    original = ShardedTrainingGenerationView._decode_payload

    def checked(self: Any, payload: bytes) -> Any:
        assert payload.startswith(b"\x28\xb5\x2f\xfd")
        assert payload != b"\x28\xb5\x2f\xfd\xe0" + b"\xff" * 8
        return original(self, payload)

    monkeypatch.setattr(ShardedTrainingGenerationView, "_decode_payload", checked)
    with _open(source, cache) as rebuilt:
        assert not rebuilt.index_reused
        assert tuple(rebuilt.iter_nodes_in_export_order()) == oracle.snapshot.nodes
    with _open(source, cache) as reused:
        assert reused.index_reused
        assert tuple(reused.iter_nodes_in_export_order()) == oracle.snapshot.nodes


def _reader_process(source: Path, cache: Path, pipe: Any, barrier: Any = None) -> None:
    """Test-only independent reader, held alive until its owner explicitly releases it."""
    if barrier is not None:
        original = ShardedTrainingGenerationView._build_index

        def synchronized(self: Any) -> None:
            barrier.wait(timeout=30)
            original(self)

        ShardedTrainingGenerationView._build_index = synchronized
    with _open(source, cache) as view:
        nodes = tuple(view.iter_nodes_in_export_order())
        pipe.send((view.index_reused, nodes))
        pipe.recv()
        assert tuple(view.iter_nodes_in_export_order()) == nodes
        view.validate_source_identity()
        pipe.send("closed")


def test_concurrent_builders_old_pins_new_generation_and_restart(
    tmp_path: Path,
) -> None:
    """Two process builders publish complete indexes; both old readers survive G2."""
    args, paths, state = setup_closed_baseline(
        tmp_path / "work", encoding="jsonl-zstd-1"
    )
    source = paths.work_dir / state.latest_tree_snapshot_path
    expected = load_closed_scientific_snapshot(source).snapshot.nodes
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    record = publication_records(state.metadata)[0]
    processes = []
    with TemporaryDirectory(dir=tmp_path, prefix="owner-") as directory:
        cache = Path(directory)
        try:
            for _ in range(2):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_reader_process, args=(source, cache, child, barrier)
                )
                process.start()
                child.close()
                processes.append((process, parent))
            for _, pipe in processes:
                assert pipe.poll(60)
                assert pipe.recv() == (False, expected)
            publish_closed_patch(paths, "next-generation")
            current = run_closed_publication(args, ClosedPublicationRunner())
            with _open(
                paths.work_dir / current.latest_tree_snapshot_path, cache
            ) as newer:
                assert newer.generation == 2
            with (
                pytest.raises(BlockingIOError),
                capsule_lease(
                    paths.work_dir,
                    record["id"],
                    version=3,
                    exclusive=True,
                    nonblocking=True,
                ),
            ):
                pass
            for process, pipe in processes:
                pipe.send("release")
                assert pipe.poll(30) and pipe.recv() == "closed"
                process.join(30)
                assert process.exitcode == 0
            # New process, same source: completed cache reused after all builders exit.
            parent, child = context.Pipe()
            process = context.Process(
                target=_reader_process, args=(source, cache, child)
            )
            process.start()
            child.close()
            processes.append((process, parent))
            assert parent.poll(60) and parent.recv() == (True, expected)
            parent.send("release")
            assert parent.poll(30) and parent.recv() == "closed"
            process.join(30)
            assert process.exitcode == 0
            assert len(list(cache.glob("v*.sqlite3"))) == 2
        finally:
            for process, pipe in processes:
                if process.is_alive():
                    process.kill()
                process.join()  # Confirm death before the owner's directory exits.
                assert not process.is_alive()
                pipe.close()
    assert not cache.exists()


def _crash_build(source: Path, cache: Path) -> None:
    """Die after private SQLite creation, before scientific validation/commit."""

    def die(*_args: Any) -> None:
        os._exit(73)

    ShardedTrainingGenerationView._build_scientific_rows = die
    _open(source, cache)


def test_owned_cache_after_process_death(tmp_path: Path) -> None:
    """A joined child is dead; neither age nor a stage timeout authorizes cleanup."""
    source = closed(tmp_path / "source", encoding="jsonl-zstd-1")
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(dir=tmp_path, prefix="owner-") as directory:
        cache = Path(directory)
        process = context.Process(target=_crash_build, args=(source, cache))
        process.start()
        try:
            process.join(60)
            assert process.exitcode == 73
            abandoned = list(cache.glob(".building-*"))
            assert abandoned and not list(cache.glob("v*.sqlite3"))
            with _open(source, cache) as rebuilt:
                assert not rebuilt.index_reused
                assert (
                    tuple(rebuilt.iter_nodes_in_export_order())
                    == load_closed_scientific_snapshot(source).snapshot.nodes
                )
            assert all(path.exists() for path in abandoned)
            with _open(source, cache) as reused:
                assert reused.index_reused
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            assert not process.is_alive()
    assert not cache.exists()


def test_killed_reader_releases_pin_but_keeps_reusable_index(tmp_path: Path) -> None:
    """The owner waits for process death before closing its private cache scope."""
    _, paths, state = setup_closed_baseline(tmp_path / "work", encoding="jsonl-zstd-1")
    source = paths.work_dir / state.latest_tree_snapshot_path
    record = publication_records(state.metadata)[0]
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(dir=tmp_path, prefix="owner-") as directory:
        cache = Path(directory)
        parent, child = context.Pipe()
        process = context.Process(target=_reader_process, args=(source, cache, child))
        process.start()
        child.close()
        try:
            assert parent.poll(60)
            reused, expected = parent.recv()
            assert not reused
            with (
                pytest.raises(BlockingIOError),
                capsule_lease(
                    paths.work_dir,
                    record["id"],
                    version=3,
                    exclusive=True,
                    nonblocking=True,
                ),
            ):
                pass
            process.kill()
            process.join(30)
            assert process.exitcode == -9
            with capsule_lease(
                paths.work_dir,
                record["id"],
                version=3,
                exclusive=True,
                nonblocking=True,
            ):
                pass
            with _open(source, cache) as reader:
                assert reader.index_reused
                assert tuple(reader.iter_nodes_in_export_order()) == expected
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            assert not process.is_alive()
            parent.close()
    assert not cache.exists()
