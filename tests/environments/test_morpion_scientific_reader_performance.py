"""A verified scientific index can lend its canonicality proof to row passes."""

import json
import sqlite3
from dataclasses import replace
from typing import TYPE_CHECKING, get_args

import pytest
import zstandard
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning.tree_to_dataset import (
    InvalidMorpionStateRefPayloadError,
    training_node_to_morpion_supervised_row,
)
from tests.environments.test_morpion_scientific_training_reader import source, view

if TYPE_CHECKING:
    from pathlib import Path


def test_only_cold_index_replays_states_across_all_dataset_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every record is canonically checked; subsequent exact row passes reuse that proof."""
    path = source(tmp_path / "source")
    oracle = load_scientific_training_generation(path)
    expected = {
        p: extract_rows_from_training_snapshot(
            args=MorpionBootstrapArgs(
                work_dir=tmp_path, dataset_family_target_policy=p
            ),
            snapshot=oracle.snapshot,
            generation=2,
        )
        for p in get_args(PvFamilyTargetPolicy)
    }
    calls = {"decode": 0, "dump": 0}
    load = MorpionStateCheckpointCodec.load_state_ref
    dump = MorpionStateCheckpointCodec.dump_state_ref

    def decoded(self: MorpionStateCheckpointCodec, payload: object) -> object:
        calls["decode"] += 1
        return load(self, payload)

    def dumped(self: MorpionStateCheckpointCodec, state: object) -> object:
        calls["dump"] += 1
        return dump(self, state)

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", decoded)
    monkeypatch.setattr(MorpionStateCheckpointCodec, "dump_state_ref", dumped)
    with view(path) as indexed:
        assert calls == {"decode": 4, "dump": 4}
        for policy, reference in expected.items():
            args = MorpionBootstrapArgs(
                work_dir=tmp_path, dataset_family_target_policy=policy
            )
            with disk_dataset_context(IndexedDatasetSource(indexed), args) as context:
                stream = context.stream(2)
                assert stream.metadata == reference.metadata
                assert tuple(stream.rows) == reference.rows
                assert tuple(context.rows()) == reference.rows
        assert calls == {"decode": 4, "dump": 4}
    with view(path) as reused:
        assert reused.index_reused
        assert calls == {"decode": 4, "dump": 4}
    # Full/legacy sources still validate through the authoritative codec.
    training_node_to_morpion_supervised_row(oracle.snapshot.nodes[0])
    assert calls["decode"] == 5


def test_corrupt_cached_reference_rebuilds_before_fast_rows(tmp_path: Path) -> None:
    """A valid SQLite file with a changed reference is not a validation certificate."""
    path = source(tmp_path / "source")
    oracle = load_scientific_training_generation(path)
    with view(path) as indexed:
        cache = indexed.index_path
    with sqlite3.connect(cache) as connection:
        row = connection.execute(
            "SELECT payload FROM records WHERE node_id='0'"
        ).fetchone()
        payload = json.loads(zstandard.ZstdDecompressor().decompress(row[0]))
        payload["state_ref_payload"] = [0, [0]]
        connection.execute(
            "UPDATE records SET payload=? WHERE node_id='0'",
            (
                zstandard.ZstdCompressor(level=3, write_checksum=True).compress(
                    json.dumps(payload).encode()
                ),
            ),
        )
    with view(path) as rebuilt:
        assert not rebuilt.index_reused
        assert tuple(rebuilt.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        args = MorpionBootstrapArgs(work_dir=tmp_path)
        expected = extract_rows_from_training_snapshot(
            args=args, snapshot=oracle.snapshot, generation=2
        )
        with disk_dataset_context(IndexedDatasetSource(rebuilt), args) as context:
            assert tuple(context.rows()) == expected.rows


def test_unverified_row_still_rejects_bad_replay(tmp_path: Path) -> None:
    """The public/default converter has no trust-by-metadata or format-version flag."""
    path = source(tmp_path / "source")
    oracle = load_scientific_training_generation(path)
    node = replace(
        oracle.snapshot.nodes[0],
        state_ref_payload=[0, [0]],
        metadata={"validated": True, "format_version": 2},
    )
    with pytest.raises(InvalidMorpionStateRefPayloadError):
        training_node_to_morpion_supervised_row(node)
