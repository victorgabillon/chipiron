"""Consumer admission and exact canonical proof reuse, including retained sources."""

import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap import (
    scientific_validation_evidence as proof,
)
from chipiron.environments.morpion.bootstrap import sharded_training_reader as reader
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    file_digest,
    publication_records,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    publish_closed_patch,
    run_closed_publication,
    setup_closed_baseline,
)

if TYPE_CHECKING:
    from pathlib import Path


def view(path: Path, cache: Path) -> reader.ShardedTrainingGenerationView:
    """Keep derived indexes outside the source and choose scientific storage."""
    return reader.ShardedTrainingGenerationView(
        path, cache_dir=cache, allow_closed_snapshot=True, allow_scientific_v2=True
    )


def rewrite_proof(paths: Any, change: Any) -> None:
    """Produce explicitly malformed fixture authority; never alter real evidence."""
    state = json.loads(paths.run_state_path.read_text())
    record = state["metadata"][PUBLICATION_KEY]["generations"][-1]
    seal = paths.work_dir / record["descriptor_path"]
    data = json.loads(seal.read_text())
    change(data)
    seal.write_text(json.dumps(data))
    record["descriptor_sha256"] = file_digest(seal)
    paths.run_state_path.write_text(json.dumps(state))


@pytest.mark.parametrize("closed", [False, True])
def test_admitted_reader_skips_only_proven_canonicality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, closed: bool
) -> None:
    """A cold managed index consumes real evidence, with exact rows and windows."""
    from tests.environments.morpion_durable_v2_helpers import setup_scientific_baseline

    setup = setup_closed_baseline if closed else setup_scientific_baseline
    _, paths, state = setup(tmp_path)
    path = paths.work_dir / state.latest_tree_snapshot_path
    with monkeypatch.context() as patch:
        patch.setattr(reader, "verified_reader_validation", lambda _: nullcontext())
        with view(path, tmp_path / "reference") as reference:
            expected = tuple(reference.iter_nodes_in_export_order())
            provenance = reference.provenance
            window = reference.select_sorted_node_window(None, 1)
    calls = []
    original = MorpionStateCheckpointCodec.load_state_ref

    def load(self: Any, payload: Any) -> Any:
        calls.append(payload)
        return original(self, payload)

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", load)
    with view(path, tmp_path / "candidate") as actual:
        assert not actual.index_reused
        assert tuple(actual.iter_nodes_in_export_order()) == expected
        assert actual.provenance == provenance
        assert actual.select_sorted_node_window(None, 1) == window
    assert calls == []
    assert proof._SCOPE.get() is None


@pytest.mark.parametrize("kind", ["absent", "stale", "descriptor"])
def test_unusable_proof_performs_complete_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Absence, changed code and nonmatching immutable descriptors grant no reuse."""
    _, paths, state = setup_closed_baseline(tmp_path)

    def change(data: Any) -> None:
        if kind == "absent":
            del data["canonical_validation"]
        elif kind == "stale":
            data["canonical_validation"]["validator"] = {"stale": True}
        else:
            data["canonical_validation"]["record_shards"][0]["sha256"] = "0" * 64

    rewrite_proof(paths, change)
    calls = []
    original = MorpionStateCheckpointCodec.load_state_ref

    def load(self: Any, payload: Any) -> Any:
        calls.append(payload)
        return original(self, payload)

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", load)
    with view(
        paths.work_dir / state.latest_tree_snapshot_path, tmp_path / "cache"
    ) as actual:
        assert len(calls) == actual.node_count()
    assert proof._SCOPE.get() is None


@pytest.mark.parametrize("kind", ["schema", "identity"])
def test_malformed_proof_fails_before_cache_use(tmp_path: Path, kind: str) -> None:
    """Even a completed derived index cannot authorize malformed current evidence."""
    _, paths, state = setup_closed_baseline(tmp_path)
    path = paths.work_dir / state.latest_tree_snapshot_path
    with view(path, tmp_path / "cache"):
        pass

    def change(data: Any) -> None:
        if kind == "schema":
            data["canonical_validation"]["unknown"] = True
        else:
            data["canonical_validation"]["scientific_identity"] = {}

    rewrite_proof(paths, change)
    with pytest.raises(
        ValueError, match=r"canonical validation evidence|identity mismatch"
    ):
        view(path, tmp_path / "cache")
    assert proof._SCOPE.get() is None


def test_retained_source_and_multiple_open_readers_do_not_leak_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Growth may advance while Dataset retains an older admitted source and lease."""
    args, paths, state = setup_closed_baseline(tmp_path)
    old = paths.work_dir / state.latest_tree_snapshot_path
    with view(old, tmp_path / "cache") as retained:
        expected = tuple(retained.iter_nodes_in_export_order())
        publish_closed_patch(paths, "advance")
        runner = ClosedPublicationRunner()
        runner.grow_delta = 1
        run_closed_publication(args, runner)
        new_state = load_bootstrap_run_state(paths.run_state_path)
        new = paths.work_dir / new_state.latest_tree_snapshot_path
        assert new != old
        with view(new, tmp_path / "cache") as current:
            assert tuple(retained.iter_nodes_in_export_order()) == expected
            assert tuple(current.iter_nodes_in_export_order()) != expected
            assert proof._SCOPE.get() is None
        with monkeypatch.context() as patch:
            patch.setattr(reader, "verified_reader_validation", lambda _: nullcontext())
            with view(new, tmp_path / "reference") as reference:
                fresh = tuple(reference.iter_nodes_in_export_order())
        with view(new, tmp_path / "another-cache") as actual:
            assert tuple(actual.iter_nodes_in_export_order()) == fresh
    # Older source remains independently admissible after closing the first view.
    with view(old, tmp_path / "other-cache") as reopened:
        assert tuple(reopened.iter_nodes_in_export_order()) == expected
    records = publication_records(new_state.metadata)
    assert records[0]["id"] != records[-1]["id"]


def test_proof_cannot_authorize_another_capsule_dependency(tmp_path: Path) -> None:
    """A managed path must select its capsule's exact scientific descriptor."""
    _, paths, state = setup_closed_baseline(tmp_path)
    root = (paths.work_dir / state.latest_tree_snapshot_path).parent
    with pytest.raises(ValueError, match="exact committed scientific dependency"):
        view(root / "records.jsonl", tmp_path / "cache")
    assert proof._SCOPE.get() is None


def test_concurrent_managed_builds_keep_proofs_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent contexts can atomically publish the same complete cache."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _, paths, state = setup_closed_baseline(tmp_path)
    path = paths.work_dir / state.latest_tree_snapshot_path
    barrier = Barrier(2)
    build = reader.ShardedTrainingGenerationView._build_index

    def synchronized(actual: Any) -> None:
        barrier.wait(timeout=20)
        build(actual)

    monkeypatch.setattr(
        reader.ShardedTrainingGenerationView, "_build_index", synchronized
    )

    def opened(_: int) -> Any:
        with view(path, tmp_path / "cache") as actual:
            nodes = tuple(actual.iter_nodes_in_export_order())
        assert proof._SCOPE.get() is None
        return nodes

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(opened, range(2)))
    assert results[0] == results[1]
    assert proof._SCOPE.get() is None
    assert not tuple((tmp_path / "cache").glob(".building-*"))


def test_dead_reader_leaves_no_pin_or_successful_partial_index(tmp_path: Path) -> None:
    """Kernel reaping releases proof and leases; recovery builds a complete view."""
    import subprocess
    import sys

    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        capsule_lease,
    )

    _, paths, state = setup_closed_baseline(tmp_path)
    path = paths.work_dir / state.latest_tree_snapshot_path
    script = """
import os,sys
from pathlib import Path
from chipiron.environments.morpion.bootstrap import sharded_training_reader as r
original=r.iter_shard_rows
def die(*args,**kwargs):
    for row in original(*args,**kwargs):
        if args[1]=='records':os._exit(73)
        yield row
r.iter_shard_rows=die
r.ShardedTrainingGenerationView(Path(sys.argv[1]),cache_dir=Path(sys.argv[2]),allow_closed_snapshot=True)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), str(tmp_path / "cache")],
        check=False,
        timeout=30,
    )
    assert child.returncode == 73
    assert not tuple((tmp_path / "cache").glob("v*.sqlite3"))
    record = publication_records(state.metadata)[-1]
    with capsule_lease(
        paths.work_dir, record["id"], exclusive=True, nonblocking=True, version=3
    ):
        pass
    with view(path, tmp_path / "cache") as recovered:
        assert not recovered.index_reused
        assert recovered.node_count() == 2
    assert proof._SCOPE.get() is None
