"""Kernel ownership, reader pins and conservative orphan collection."""

import hashlib
import json
import os
import select
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import chipiron.environments.morpion.bootstrap.generation_publication as publication
from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    ATTEMPT_MARKER,
    PUBLICATION_KEY,
    PUBLICATION_NAMESPACE,
    capsule_lease,
    capsule_path,
    publication_records,
    publish_directory_if_absent,
    read_object,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import (
    MODEL,
    fixture_patch,
    publish_fixture_patch,
    save_fixture_cursor,
)


def _line(process: subprocess.Popen[str]) -> str:
    """Use bounded barriers, with no timing-dependent race assumptions."""
    assert process.stdout is not None
    assert select.select([process.stdout], [], [], 20)[0], "Child barrier timed out"
    return process.stdout.readline().strip()


def test_reader_continues_while_competing_writer_is_rejected(tmp_path: Path) -> None:
    """Growth ownership never blocks independent readers of an older capsule."""
    args, paths = setup_publication_baseline(tmp_path)
    baseline = load_bootstrap_run_state(paths.run_state_path)
    record = publication_records(baseline.metadata)[0]
    root = capsule_path(tmp_path, record["id"])
    snapshot = root / "tree_exports_sharded/generation_000001.json"
    old_nodes = load_morpion_sharded_training_tree_snapshot(snapshot).nodes
    publish_fixture_patch(paths)
    script = """
import sys
from pathlib import Path
from dataclasses import replace
import chipiron.environments.morpion.bootstrap.generation_publication as publication
from tests.environments.morpion_generation_publication_helpers import PublicationRunner,run_publication
from tests.environments.morpion_patch_receipt_helpers import receipt_args
root=Path(sys.argv[1])
args=replace(receipt_args(root),training_export_mode='sharded',growth_skip_training_export=False)
original=publication.publish_directory_if_absent
def publish(staging,final,work_dir):
    if final.parent.name == 'objects':
        print('owned',flush=True)
        assert sys.stdin.readline().strip() == 'continue'
    original(staging,final,work_dir)
publication.publish_directory_if_absent=publish
runner=PublicationRunner()
state=run_publication(args,runner)
print(f'committed:{state.generation}:{runner.apply_calls}',flush=True)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as writer:
        try:
            assert _line(writer) == "owned"
            loser = PublicationRunner()
            with pytest.raises(BlockingIOError):
                run_publication(args, loser)
            assert loser.apply_calls == 0
            with ShardedTrainingGenerationView(
                snapshot, cache_dir=tmp_path / "cache"
            ) as reader:
                assert tuple(reader.iter_nodes_in_export_order()) == old_nodes
                assert writer.stdin is not None
                writer.stdin.write("continue\n")
                writer.stdin.flush()
                assert _line(writer) == "committed:2:1"
                assert tuple(reader.iter_nodes_in_export_order()) == old_nodes
                assert writer.wait(timeout=10) == 0
            state = load_bootstrap_run_state(paths.run_state_path)
            assert (
                state.generation == 2 and len(publication_records(state.metadata)) == 2
            )
            assert not paths.pipeline_reevaluation_patch_path.exists()
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait(timeout=10)


def test_directory_no_replace_preserves_winner(tmp_path: Path) -> None:
    """The atomic primitive never overwrites an existing complete directory."""
    staging = tmp_path / "staging"
    objects = tmp_path / "objects"
    staging.mkdir()
    objects.mkdir()
    first, second, final = staging / "a", staging / "b", objects / "winner"
    first.mkdir()
    second.mkdir()
    (first / "payload").write_bytes(b"first")
    (second / "payload").write_bytes(b"second")
    publish_directory_if_absent(first, final, tmp_path)
    with pytest.raises(FileExistsError):
        publish_directory_if_absent(second, final, tmp_path)
    assert (final / "payload").read_bytes() == b"first"
    assert (second / "payload").read_bytes() == b"second"


def test_hardlink_metadata_changes_do_not_invalidate_immutable_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Managed readers use committed byte hashes when a new link changes inode ctime."""
    _args, paths = setup_publication_baseline(tmp_path)
    state = load_bootstrap_run_state(paths.run_state_path)
    root = capsule_path(tmp_path, publication_records(state.metadata)[0]["id"])
    snapshot = root / "tree_exports_sharded/generation_000001.json"
    original_identity = ShardedTrainingGenerationView._source_identity
    original_digest = hashlib.file_digest
    fingerprinting = False
    linked = False

    def identity(view: ShardedTrainingGenerationView) -> str:
        nonlocal fingerprinting
        fingerprinting = True
        try:
            return original_identity(view)
        finally:
            fingerprinting = False

    def digest(handle: object, algorithm: str) -> object:
        nonlocal linked
        result = original_digest(handle, algorithm)
        path = Path(handle.name)
        if fingerprinting and not linked and path.parent.name == "node_shards":
            os.link(path, tmp_path / "new_immutable_link")
            linked = True
        return result

    monkeypatch.setattr(ShardedTrainingGenerationView, "_source_identity", identity)
    monkeypatch.setattr(hashlib, "file_digest", digest)
    with ShardedTrainingGenerationView(
        snapshot, cache_dir=tmp_path / "cache"
    ) as reader:
        assert len(tuple(reader.iter_nodes_in_export_order())) == 2
    assert linked


def test_retention_pins_all_committed_provenance_and_respects_live_pins(
    tmp_path: Path,
) -> None:
    """Only owned uncommitted attempts are collectible; committed roots are permanent."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    first = run_publication(args, PublicationRunner())
    publish_fixture_patch(paths, fixture_patch("patch-2"))
    state = run_publication(args, PublicationRunner())
    assert state.generation == 3
    assert state.latest_runtime_checkpoint_path is not None
    assert json.loads(
        (tmp_path / state.latest_runtime_checkpoint_path).read_bytes()
    ) == {"root": 8.0, "leaf": 10.0}
    roots = publication.DurableGenerationPublication(paths).retention_roots(state)
    for record in publication_records(state.metadata):
        assert capsule_path(tmp_path, record["id"]) in roots
        validate_capsule(tmp_path, record)
    assert tmp_path / MODEL in roots and paths.sharded_tree_snapshot_dir in roots
    assert tmp_path / first.latest_runtime_checkpoint_path in roots
    identifier = uuid.uuid4().hex
    orphan = tmp_path / PUBLICATION_NAMESPACE / "staging" / identifier
    orphan.mkdir()
    atomic_json(
        orphan / ATTEMPT_MARKER, {"protocol": PUBLICATION_KEY, "id": identifier}
    )
    (orphan / "payload").write_bytes(b"uncommitted")
    with capsule_lease(tmp_path, identifier):
        assert publication.prune_uncommitted_generation_artifacts(paths) == []
        assert orphan.exists()
    assert publication.prune_uncommitted_generation_artifacts(paths) == [identifier]
    assert not orphan.exists()
    assert (tmp_path / PUBLICATION_NAMESPACE / "pins" / f"{identifier}.lock").is_file()
    for record in publication_records(state.metadata):
        validate_capsule(tmp_path, record)


def test_committed_receipt_uses_frozen_original_model_after_source_disappears(
    tmp_path: Path,
) -> None:
    """Model provenance is the pinned bytes, never a later mutable bundle alias."""
    args, paths = setup_publication_baseline(tmp_path)
    patch = publish_fixture_patch(paths, cursor=False)
    state = run_publication(args, PublicationRunner())
    (tmp_path / MODEL / "param.pt").unlink()
    save_fixture_cursor(paths, patch)
    runner = PublicationRunner()
    recovered = run_publication(args, runner, max_cycles=0)
    assert runner.apply_calls == 0 and recovered.generation == state.generation
    assert not paths.pipeline_reevaluation_patch_path.exists()
    descriptor = read_object(
        tmp_path / publication_records(state.metadata)[-1]["descriptor_path"]
    )
    assert descriptor["model_pins"][0]["origin"] == MODEL


def test_retention_requires_growth_ownership(tmp_path: Path) -> None:
    """A collector cannot race a writer or another collector, even between commits."""
    _args, paths = setup_publication_baseline(tmp_path)
    with growth_receipt_ownership(paths), pytest.raises(BlockingIOError):
        publication.prune_uncommitted_generation_artifacts(paths)
