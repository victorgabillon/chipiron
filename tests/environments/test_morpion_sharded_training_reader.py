"""Differential and resource tests for the bounded published-generation view."""
# ruff: noqa: TRY003

from __future__ import annotations

import json
import os
import random
import tracemalloc
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

import chipiron.environments.morpion.bootstrap.sharded_training_reader as reader_module
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    select_reevaluation_node_window,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
    iter_shard_rows,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from anemone.training_export import TrainingNodeSnapshot

    from chipiron.environments.morpion.bootstrap.sharded_training_export import (
        MorpionShardedTrainingNodeRecord,
        MorpionShardedTrainingNodeUpdate,
    )


def write_generation(
    root: Path,
    ids: Sequence[str],
    *,
    generation: int = 1,
    new_ids: Sequence[str] | None = None,
) -> Path:
    """Write small synthetic documents with diverse opaque payloads and updates."""
    root.mkdir(parents=True, exist_ok=True)
    new_ids = ids if new_ids is None else new_ids
    refs: tuple[object, ...] = (
        {"kind": "anchor", "anchor_ref": [0, [2, 8]]},
        (0, (4, 8)),
        {"opaque": {"large_integer": 2**100, "float": 1.25, "unicode": "é"}},
        None,
        {"kind": "delta", "state_parent_node_id": 1, "delta_ref": 7},
    )
    (root / f"nodes-{generation}.json").write_text(
        json.dumps({
            "nodes": [
                {
                    "node_id": node_id,
                    "parent_ids": [ids[0]] if i else [],
                    "depth": i,
                    "creation_generation": generation,
                    "state_ref_payload": refs[i % len(refs)],
                }
                for i, node_id in enumerate(new_ids)
            ]
        }),
        encoding="utf-8",
    )
    (root / f"updates-{generation}.json").write_text(
        json.dumps({
            "updates": [
                {
                    "node_id": node_id,
                    "order_index": (len(ids) - i) // 2,
                    "child_ids": list(ids[i + 1 : i + 2])
                    if generation == 1
                    else list(ids[-2:]),
                    "direct_value_scalar": None
                    if i % 3 == 0
                    else i + 0.25 * generation,
                    "backed_up_value_scalar": None
                    if i % 2 == 0
                    else i + 0.5 * generation,
                    "is_terminal": i % 2 == 0,
                    "is_exact": i % 3 == 1,
                    "over_event_label": "win" if i % 2 == 0 else None,
                    "visit_count": i * generation,
                    "metadata": {
                        "generation": generation,
                        "fraction": 0.1,
                        "huge": 2**100,
                    },
                }
                for i, node_id in enumerate(ids)
            ]
        }),
        encoding="utf-8",
    )
    manifest_path = root / f"generation_{generation:06d}.json"
    manifest_path.write_text(
        json.dumps({
            "generation": generation,
            "root_node_id": ids[0] if ids else "empty",
            "node_count": len(ids),
            "new_node_count": len(new_ids),
            "node_shard_path": f"nodes-{generation}.json",
            "update_shard_path": f"updates-{generation}.json",
        }),
        encoding="utf-8",
    )
    root_manifest = root / "manifest.json"
    manifests: dict[str, Any] = (
        json.loads(root_manifest.read_text())
        if root_manifest.exists()
        else {"generation_manifests": {}}
    )
    manifests["generation_manifests"][str(generation)] = manifest_path.name
    manifests["latest_generation"] = generation
    root_manifest.write_text(json.dumps(manifests), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("generation", [1, 2, 3])
def test_every_snapshot_field_and_export_order_match(
    tmp_path: Path, generation: int
) -> None:
    """New/reused records and changed mutable links/values match every oracle field."""
    ids = ["1", "10", "100", "2", "20", "3"]
    path = write_generation(tmp_path / "export", ids)
    for current in range(2, generation + 1):
        ids.append(f"new-{current}")
        path = write_generation(path.parent, ids, generation=current, new_ids=[ids[-1]])
    oracle = load_morpion_sharded_training_tree_snapshot(path)
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.generation == generation
        assert view.root_node_id == oracle.root_node_id
        assert view.node_count() == len(oracle.nodes)
        assert [asdict(node) for node in view.iter_nodes_in_export_order()] == [
            asdict(node) for node in oracle.nodes
        ]
        by_id = {node.node_id: node for node in oracle.nodes}
        selected = tuple(reversed(ids[::2]))
        assert tuple(asdict(node) for node in view.load_nodes(selected)) == tuple(
            asdict(by_id[node_id]) for node_id in selected
        )
        with pytest.raises(KeyError):
            view.load_nodes(["absent"])


@pytest.mark.exhaustive
def test_randomized_cursor_parity_and_unicode_collation(tmp_path: Path) -> None:
    """Thousands of edge cases compare directly with the unchanged legacy selector."""
    rng = random.Random(1427)
    population = [str(i) for i in range(50)] + [
        "",
        "\0",
        "a\0b",
        "A",
        "a",
        "é",
        "中",
        "\uffff",
        "😀",
        "\U0010ffff",
    ]
    for trial in range(40):
        ids = rng.sample(population, rng.randrange(len(population)))
        path = write_generation(tmp_path / str(trial), ids)
        ordered = tuple(sorted(ids))
        with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
            for cursor in [None, "missing", *ids]:
                for size in {0, 1, 2, len(ids), len(ids) + 1, rng.randrange(100)}:
                    assert view.select_sorted_node_window(
                        cursor, size
                    ) == select_reevaluation_node_window(
                        ordered, start_cursor=cursor, max_nodes=size
                    )


def test_reuse_content_invalidation_and_source_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even same-size content edits with restored mtime invalidate a complete cache."""
    path = write_generation(tmp_path / "source", ["1", "2"])
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as first:
        assert not first.index_reused
        first_index = first.index_path
        assert first.index_bytes > 0
    with monkeypatch.context() as patch:
        patch.setattr(
            reader_module,
            "iter_shard_rows",
            lambda *_: pytest.fail("parsed a shard during reuse"),
        )
        with ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "cache"
        ) as second:
            assert second.index_reused
            assert second.index_path == first_index
    assert before == {p: p.read_bytes() for p in path.parent.iterdir()}
    update_path = path.parent / "updates-1.json"
    info = update_path.stat()
    update_path.write_bytes(update_path.read_bytes().replace(b"0.25", b"0.75"))
    # The fixture uses 1.25; edit a field without changing byte count.
    update_path.write_bytes(update_path.read_bytes().replace(b"1.25", b"1.75"))
    os.utime(update_path, ns=(info.st_atime_ns, info.st_mtime_ns))
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as changed:
        assert not changed.index_reused
        assert changed.index_path != first_index
        assert changed.load_nodes(["2"])[0].direct_value_scalar == 1.75
    node_path = path.parent / "nodes-1.json"
    node_path.write_text(node_path.read_text().replace('"depth": 1', '"depth": 9'))
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as changed:
        assert not changed.index_reused
        assert changed.load_nodes(["2"])[0].depth == 9
    payload = json.loads(path.read_text())
    payload["root_node_id"] = "2"
    path.write_text(json.dumps(payload))
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as changed:
        assert not changed.index_reused
        assert changed.root_node_id == "2"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "truncated",
        "wrong_array",
        "missing_array",
        "duplicate_update",
        "duplicate_record",
        "orphan",
        "count",
        "root",
        "version",
    ],
)
def test_malformed_artifacts_fail_and_do_not_publish(
    tmp_path: Path, damage: str
) -> None:
    """Malformed artifacts are rejected explicitly, with no cache marked complete."""
    path = write_generation(tmp_path / "source", ["1", "2"])
    updates = path.parent / "updates-1.json"
    if damage == "missing":
        updates.unlink()
    elif damage == "truncated":
        updates.write_text(updates.read_text()[:-3])
    elif damage == "wrong_array":
        updates.write_text('{"updates": {}}')
    elif damage == "missing_array":
        updates.write_text("{}")
    elif damage in {"duplicate_update", "orphan"}:
        payload = json.loads(updates.read_text())
        payload["updates"][1]["node_id"] = (
            "1" if damage == "duplicate_update" else "absent"
        )
        updates.write_text(json.dumps(payload))
    elif damage == "duplicate_record":
        nodes = path.parent / "nodes-1.json"
        payload = json.loads(nodes.read_text())
        payload["nodes"].append(payload["nodes"][0])
        nodes.write_text(json.dumps(payload))
    else:
        payload = json.loads(path.read_text())
        payload[
            {
                "count": "node_count",
                "root": "root_node_id",
                "version": "format_version",
            }[damage]
        ] = 99
        path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileNotFoundError)):
        ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))


def test_interrupted_build_and_corrupt_cache_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed prefixes and abandoned temporary files never pass as completed indexes."""
    path = write_generation(tmp_path / "source", [str(i) for i in range(300)])
    original = reader_module.iter_shard_rows

    def interrupted(shard: Path, field: str) -> Iterator[dict[str, object]]:
        for i, row in enumerate(original(shard, field)):
            if i == 270:
                raise InterruptedError(
                    "synthetic interruption after bounded transaction"
                )
            yield row

    with monkeypatch.context() as patch:
        patch.setattr(reader_module, "iter_shard_rows", interrupted)
        with pytest.raises(InterruptedError):
            ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))
    abandoned = tmp_path / "cache" / ".building-abandoned.sqlite3"
    abandoned.write_bytes(b"interrupted process")
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert not view.index_reused
        index_path = view.index_path
    index_path.write_bytes(b"broken sqlite")
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert not view.index_reused
        assert view.node_count() == 300


def test_parser_buffers_do_not_scale_with_array_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure the parser alone; forbid stdlib whole-value loads of shard arrays."""
    peaks: list[int] = []
    for count in (100, 10000):
        path = tmp_path / f"{count}.json"
        row = json.dumps({"node_id": "1", "state_ref_payload": {"opaque": "x" * 2048}})
        with path.open("w") as stream:
            stream.write('{"nodes":[')
            for index in range(count):
                stream.write(("," if index else "") + row)
            stream.write("]}")
        with monkeypatch.context() as patch:
            patch.setattr(
                json, "load", lambda *_a, **_kw: pytest.fail("whole-file JSON load")
            )
            patch.setattr(
                json, "loads", lambda *_a, **_kw: pytest.fail("whole-value JSON load")
            )
            tracemalloc.start()
            assert sum(1 for _ in iter_shard_rows(path, "nodes")) == count
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        peaks.append(peak)
    assert max(peaks) < 2 * 1024 * 1024
    assert peaks[1] < peaks[0] + 256 * 1024


def test_cache_cannot_be_written_into_source(tmp_path: Path) -> None:
    """An explicit bad cache override cannot mutate the source export."""
    path = write_generation(tmp_path / "source", [])
    with pytest.raises(ValueError, match="outside"):
        ShardedTrainingGenerationView(path, cache_dir=path.parent / "cache")


def test_index_build_never_merges_nodes_or_loads_raw_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index construction validates rows without allocating snapshots or raw arrays."""
    path = write_generation(tmp_path / "source", [str(i) for i in range(500)])
    original_loads = json.loads
    original_merge = reader_module._merge_node_record_and_update
    merged: list[str] = []

    def bounded_loads(text: str) -> object:
        assert len(text) < 4096, "whole shard passed to json.loads"
        return original_loads(text)

    def merge(
        record: MorpionShardedTrainingNodeRecord,
        update: MorpionShardedTrainingNodeUpdate,
    ) -> TrainingNodeSnapshot:
        merged.append(record.node_id)
        return original_merge(record, update)

    monkeypatch.setattr(json, "loads", bounded_loads)
    monkeypatch.setattr(reader_module, "_merge_node_record_and_update", merge)
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert merged == []
        assert len(view.load_nodes(["1", "2"])) == 2
        assert merged == ["1", "2"]


def test_index_build_python_memory_is_bounded(tmp_path: Path) -> None:
    """Measure indexing separately from fixture and parser-only allocation peaks."""
    peaks: list[int] = []
    for count in (100, 3000):
        path = write_generation(tmp_path / str(count), [str(i) for i in range(count)])
        tracemalloc.start()
        with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
            assert view.node_count() == count
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    assert max(peaks) < 4 * 1024 * 1024
    assert peaks[1] < peaks[0] + 1024 * 1024


def test_sources_changed_during_build_do_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutation between fingerprint and commit cannot publish a reusable index."""
    path = write_generation(tmp_path / "source", ["1", "2"])
    original = reader_module.iter_shard_rows

    def mutate(shard: Path, field: str) -> Iterator[dict[str, object]]:
        yield from original(shard, field)
        if field == "updates":
            shard.write_bytes(shard.read_bytes() + b"\n")

    with monkeypatch.context() as patch:
        patch.setattr(reader_module, "iter_shard_rows", mutate)
        with pytest.raises(ValueError, match="changed during"):
            ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.node_count() == 2


def test_concurrent_cache_builders_publish_complete_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two synthetic builders may race, but neither exposes a partial final index."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = write_generation(tmp_path / "source", [str(i) for i in range(300)])
    barrier = Barrier(2)
    valid_index = ShardedTrainingGenerationView._valid_index

    def both_cold(view: ShardedTrainingGenerationView) -> bool:
        assert not valid_index(view)
        barrier.wait(timeout=10)
        return False

    def build(_: int) -> tuple[int, Path]:
        with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
            return view.node_count(), view.index_path

    with monkeypatch.context() as patch:
        patch.setattr(ShardedTrainingGenerationView, "_valid_index", both_cold)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(build, (1, 2)))
    assert results[0] == results[1]
    assert results[0][0] == 300
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as reused:
        assert reused.index_reused
        assert len(reused.load_nodes(["1", "20"])) == 2


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_unpaired_surrogate_ids_fail_without_silent_replacement(
    tmp_path: Path, surrogate: str
) -> None:
    """C-backend string replacement must never silently change node identity."""
    path = write_generation(tmp_path / "source", [surrogate])
    with pytest.raises(ValueError, match="Unicode scalar"):
        ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache")


def test_opaque_surrogate_escapes_and_chunk_boundaries_are_lossless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opaque refs retain even stdlib surrogate strings; astral IDs stay exact."""
    path = write_generation(tmp_path / "source", ["😀"])
    shard = path.parent / "nodes-1.json"
    payload = json.loads(shard.read_text())
    payload["nodes"][0]["state_ref_payload"] = {
        "opaque": "\ud800",
        "literal": "\\udfff",
    }
    shard.write_text(json.dumps(payload))
    monkeypatch.setattr(reader_module, "PARSER_BUFFER_BYTES", 7)
    oracle = load_morpion_sharded_training_tree_snapshot(path)
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.load_nodes(["😀"]) == oracle.nodes
