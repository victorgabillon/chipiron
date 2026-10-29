"""Immutable scientific v1 generations and checkpoint-bound receipt recovery."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from anemone.training_export import load_training_tree_snapshot

import chipiron.environments.morpion.bootstrap.sharded_training_export as v1
from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    prune_generation_files,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    CAPSULE_SEAL,
    PUBLICATION_KEY,
    PUBLICATION_NAMESPACE,
    GenerationPublicationError,
    artifact_digest,
    capsule_path,
    publication_records,
    read_object,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
    write_v1_export,
)
from tests.environments.morpion_patch_receipt_helpers import (
    BASELINE,
    MODEL,
    PATCHED,
    fixture_patch,
    publish_fixture_patch,
    setup_receipt_baseline,
)


def _all_bytes(root: Path) -> dict[str, bytes]:
    """Capture payload parity, including catalog/index and scientific metadata."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_adoption_ignores_ahead_legacy_index_without_modifying_legacy_files(
    tmp_path: Path,
) -> None:
    """A root catalog/index ahead of run state cannot suppress retry creation records."""
    args, paths = setup_publication_baseline(tmp_path, adopt_publication=False)
    write_v1_export({**BASELINE, "new": 666.0}, paths.sharded_tree_snapshot_dir, 2)
    legacy_before = _all_bytes(paths.sharded_tree_snapshot_dir)
    run_publication(args, PublicationRunner(), max_cycles=0)
    state = load_bootstrap_run_state(paths.run_state_path)
    baseline = capsule_path(tmp_path, publication_records(state.metadata)[0]["id"])
    private_index = read_object(baseline / "tree_exports_sharded/node_index.json")
    assert private_index["node_id_to_creation_generation"] == {"root": 1, "leaf": 1}
    publish_fixture_patch(paths)
    runner = PublicationRunner()
    runner.grow_delta = 1
    state = run_publication(args, runner)
    assert state.latest_tree_snapshot_path is not None
    tree = tmp_path / state.latest_tree_snapshot_path
    created = read_object(tree.parent / "node_shards/generation_000002.json")["nodes"]
    assert [node["node_id"] for node in created] == ["new"]
    assert _all_bytes(paths.sharded_tree_snapshot_dir) == legacy_before


@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
def test_every_dependency_is_synced_before_commit_and_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_format: str
) -> None:
    """Trace all files and new ancestor entries across capsule/state publication."""
    import chipiron.environments.morpion.bootstrap.generation_publication as publication

    args, paths = setup_publication_baseline(tmp_path)
    args = replace(args, runtime_checkpoint_format=checkpoint_format)
    publish_fixture_patch(paths)
    actual_sync, actual_publish, actual_replace, actual_unlink = (
        os.fsync,
        publication.publish_directory_if_absent,
        os.replace,
        Path.unlink,
    )
    events: list[tuple[str, str]] = []
    staged_root: Path | None = None

    def sync(descriptor: int) -> None:
        actual_sync(descriptor)
        events.append(("sync", os.readlink(f"/proc/self/fd/{descriptor}")))

    def publish(staging: Path, final: Path, root: Path) -> None:
        nonlocal staged_root
        if final.parent.name == "objects":
            staged_root = staging
        events.append(("publish", str(final)))
        actual_publish(staging, final, root)
        events.append(("durable", str(final)))

    def replace_state(source: str | Path, destination: str | Path) -> None:
        actual_replace(source, destination)
        events.append(("replace", str(destination)))

    def unlink(path: Path, missing_ok: bool = False) -> None:
        actual_unlink(path, missing_ok=missing_ok)
        events.append(("unlink", str(path)))

    with monkeypatch.context() as trace:
        trace.setattr(os, "fsync", sync)
        trace.setattr(publication, "publish_directory_if_absent", publish)
        trace.setattr(os, "replace", replace_state)
        trace.setattr(Path, "unlink", unlink)
        state = run_publication(args, PublicationRunner())
    record = publication_records(state.metadata)[-1]
    final = capsule_path(tmp_path, record["id"])
    descriptor = read_object(final / CAPSULE_SEAL)
    publish_index = events.index(("publish", str(final)))
    durable_index = events.index(("durable", str(final)))
    replace_index = events.index(("replace", str(paths.run_state_path)))
    commit_index = events.index(("sync", str(tmp_path)), replace_index)
    ack_index = events.index(("unlink", str(paths.pipeline_reevaluation_patch_path)))
    assert publish_index < durable_index < replace_index < commit_index < ack_index
    assert staged_root is not None
    for entry in descriptor["files"]:
        path = staged_root / entry["path"]
        assert ("sync", str(path)) in events[:publish_index]
        assert ("sync", str(path.parent)) in events[:publish_index]
    assert ("sync", str(staged_root / CAPSULE_SEAL)) in events[:publish_index]
    assert ("sync", str(final.parent)) in events[publish_index:durable_index]
    assert ("sync", str(staged_root.parent)) in events[publish_index:durable_index]
    assert ("sync", str(paths.pipeline_reevaluation_cursor_path)) in events[
        commit_index:ack_index
    ]
    assert ("sync", str(paths.pipeline_dir)) in events[ack_index + 1 :]


@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
def test_checkpoint_bytes_match_original_writer(
    tmp_path: Path, checkpoint_format: str
) -> None:
    """Publication never changes checkpoint payload bytes or codec-selection basenames."""
    args, paths = setup_publication_baseline(tmp_path)
    args = replace(args, runtime_checkpoint_format=checkpoint_format)
    publish_fixture_patch(paths)
    state = run_publication(args, PublicationRunner())
    assert state.latest_runtime_checkpoint_path is not None
    actual = tmp_path / state.latest_runtime_checkpoint_path
    control = PublicationRunner()
    control.values = dict(PATCHED)
    expected = tmp_path / "control" / actual.name
    control.save_checkpoint(expected)
    if actual.is_file():
        assert actual.read_bytes() == expected.read_bytes()
    else:
        assert _all_bytes(actual) == _all_bytes(expected)


def test_new_patch_during_ack_survives_retention_and_applies_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new singleton publication cannot be deleted using the previous receipt."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    original_unlink = Path.unlink
    published_new = False

    def unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal published_new
        original_unlink(path, missing_ok=missing_ok)
        if path == paths.pipeline_reevaluation_patch_path and not published_new:
            published_new = True
            publish_fixture_patch(paths, fixture_patch("patch-2"))

    with monkeypatch.context() as producer:
        producer.setattr(Path, "unlink", unlink)
        first = run_publication(args, PublicationRunner())
    assert published_new and paths.pipeline_reevaluation_patch_path.is_file()
    assert set(receipt_ledger(first)) == {"patch-1"}
    run_publication(args, PublicationRunner(), max_cycles=0)
    assert paths.pipeline_reevaluation_patch_path.is_file()
    runner = PublicationRunner()
    second = run_publication(args, runner)
    assert runner.apply_calls == 1 and second.generation == 3
    assert second.latest_runtime_checkpoint_path is not None
    assert json.loads(
        (tmp_path / second.latest_runtime_checkpoint_path).read_bytes()
    ) == {"root": 8.0, "leaf": 10.0}
    assert set(receipt_ledger(second)) == {"patch-1", "patch-2"}


@pytest.mark.parametrize(
    "relative",
    [
        "manifest.json",
        "generation_000001.json",
        "node_shards/generation_000001.json",
        "update_shards/generation_000001.json",
    ],
)
def test_scientific_v2_headers_are_never_partially_adopted(
    tmp_path: Path, relative: str
) -> None:
    """Publication protocol v1 must not silently parse/normalize scientific v2 data."""
    args, paths = setup_publication_baseline(tmp_path, adopt_publication=False)
    path = paths.sharded_tree_snapshot_dir / relative
    payload = read_object(path)
    payload["format_version"] = 2
    path.write_text(json.dumps(payload, sort_keys=True))
    before = path.read_bytes()
    with pytest.raises(GenerationPublicationError, match="scientific"):
        run_publication(args, PublicationRunner(), max_cycles=0)
    assert path.read_bytes() == before
    assert (
        PUBLICATION_KEY not in load_bootstrap_run_state(paths.run_state_path).metadata
    )


def test_checkpoint_only_generation_preserves_sharded_source_history(
    tmp_path: Path,
) -> None:
    """Skipped exports remain absent; later v1 catalogs preserve their generation gaps."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    first = run_publication(
        replace(args, growth_skip_training_export=True), PublicationRunner()
    )
    assert first.generation == 2 and first.latest_tree_snapshot_path is None
    publish_fixture_patch(paths, fixture_patch("patch-2"))
    second = run_publication(args, PublicationRunner())
    assert second.generation == 3 and second.latest_tree_snapshot_path is not None
    catalog = read_object(
        (tmp_path / second.latest_tree_snapshot_path).parent / "manifest.json"
    )
    assert set(catalog["generation_manifests"]) == {"1", "3"}
    assert len(publication_records(second.metadata)) == 3
    assert set(receipt_ledger(second)) == {"patch-1", "patch-2"}


def test_corrupt_config_copy_fails_before_patch_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generation's immutable config must match the accepted scientific hash."""
    import chipiron.environments.morpion.bootstrap.generation_publication as publication

    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    original = publication._copy_artifact

    def copy(source: Path, destination: Path, *, link: bool = False) -> None:
        original(source, destination, link=link)
        if source == paths.bootstrap_config_path:
            destination.write_text("{}")

    monkeypatch.setattr(publication, "_copy_artifact", copy)
    runner = PublicationRunner()
    with pytest.raises(GenerationPublicationError, match="config identity"):
        run_publication(args, runner)
    assert runner.apply_calls == 0
    assert paths.pipeline_reevaluation_patch_path.exists()
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1


def test_pr79_receipts_survive_adoption_without_rewriting_checkpoint_identity(
    tmp_path: Path,
) -> None:
    """Adopt a real PR #79 commit/ledger before enabling sharded publication."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    before = run_pipeline_growth_stage(
        args, PublicationRunner(), checkpoint_bound_receipts=True
    )
    original_receipts = dict(receipt_ledger(before))
    original_checkpoint = tmp_path / before.latest_runtime_checkpoint_path
    original_bytes = original_checkpoint.read_bytes()
    adopted = run_publication(args, PublicationRunner(), max_cycles=0)
    assert (
        adopted.latest_runtime_checkpoint_path == before.latest_runtime_checkpoint_path
    )
    assert receipt_ledger(adopted) == original_receipts
    assert original_checkpoint.read_bytes() == original_bytes
    args = replace(
        args, training_export_mode="sharded", growth_skip_training_export=False
    )
    publish_fixture_patch(paths, fixture_patch("patch-2"))
    after = run_publication(args, PublicationRunner())
    assert after.generation == 3 and after.latest_runtime_checkpoint_path is not None
    assert json.loads(
        (tmp_path / after.latest_runtime_checkpoint_path).read_bytes()
    ) == {"root": 8.0, "leaf": 10.0}
    assert receipt_ledger(after)["patch-1"] == original_receipts["patch-1"]
    assert original_checkpoint.read_bytes() == original_bytes
    publish_fixture_patch(paths, fixture_patch())
    runner = PublicationRunner()
    run_publication(args, runner, max_cycles=0)
    assert (
        runner.apply_calls == 0 and not paths.pipeline_reevaluation_patch_path.exists()
    )


def test_v1_byte_parity_and_immutable_prefix(tmp_path: Path) -> None:
    """Only publication paths change; every scientific v1 file byte matches control."""
    args, paths = setup_publication_baseline(tmp_path / "durable")
    legacy_bytes = _all_bytes(paths.sharded_tree_snapshot_dir)
    checkpoint_before = (
        paths.runtime_checkpoint_dir / "generation_000001.json.zst"
    ).read_bytes()
    old_state = load_bootstrap_run_state(paths.run_state_path)
    old_record = publication_records(old_state.metadata)[-1]
    old_capsule = capsule_path(paths.work_dir, old_record["id"])
    old_bytes = _all_bytes(old_capsule)
    control = tmp_path / "control"
    write_v1_export(BASELINE, control, 1)
    publish_fixture_patch(paths)
    runner = PublicationRunner()
    runner.grow_delta = 1
    state = run_publication(args, runner)
    assert state.generation == 2 and runner.apply_calls == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert state.latest_tree_snapshot_path is not None
    exported = paths.work_dir / state.latest_tree_snapshot_path
    values = {**PATCHED, "new": 0.0}
    write_v1_export(values, control, 2)
    assert _all_bytes(exported.parent) == _all_bytes(control)
    assert _all_bytes(old_capsule) == old_bytes
    old_export = old_capsule / "tree_exports_sharded"
    assert (old_export / "node_shards/generation_000001.json").stat().st_ino == (
        exported.parent / "node_shards/generation_000001.json"
    ).stat().st_ino
    assert (old_export / "manifest.json").stat().st_ino != (
        exported.parent / "manifest.json"
    ).stat().st_ino
    assert (old_export / "node_index.json").stat().st_ino != (
        exported.parent / "node_index.json"
    ).stat().st_ino
    assert _all_bytes(paths.sharded_tree_snapshot_dir) == legacy_bytes
    assert (
        paths.runtime_checkpoint_dir / "generation_000001.json.zst"
    ).read_bytes() == checkpoint_before
    snapshot = v1.load_morpion_sharded_training_tree_snapshot(exported)
    with ShardedTrainingGenerationView(exported, cache_dir=tmp_path / "cache") as view:
        assert tuple(view.iter_nodes_in_export_order()) == snapshot.nodes
    descriptor = validate_capsule(
        paths.work_dir, publication_records(state.metadata)[-1]
    )
    assert descriptor["protocol"] == PUBLICATION_KEY
    assert snapshot.metadata["source_generation"] == 2
    assert (
        json.loads(
            (exported.parent / "node_shards/generation_000002.json").read_bytes()
        )["nodes"][0]["creation_generation"]
        == 2
    )
    assert receipt_ledger(state)["patch-1"].tree_generation == 1
    model = descriptor["model_pins"][0]
    frozen_model = exported.parent.parent / model["path"]
    assert model["origin"] == MODEL and artifact_digest(
        frozen_model
    ) == artifact_digest(paths.work_dir / MODEL)
    assert len(publication_records(state.metadata)) == 2


def test_advanced_abandoned_index_cannot_omit_creation_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce the audited retry boundary, then recover solely from committed roots."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    original_write = v1._write_json
    baseline = load_bootstrap_run_state(paths.run_state_path)

    def write_then_fail(path: Path, payload: dict[str, object]) -> object:
        result = original_write(path, payload)
        if path.name == "node_index.json" and "new" in payload.get(
            "node_id_to_creation_generation", {}
        ):
            message = "crash after advanced private index"
            raise OSError(message)
        return result

    first = PublicationRunner()
    first.grow_delta = 1
    with monkeypatch.context() as fault:
        fault.setattr(v1, "_write_json", write_then_fail)
        with pytest.raises(OSError, match="advanced private index"):
            run_publication(args, first)
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1
    assert paths.pipeline_reevaluation_patch_path.exists()
    old_export = paths.work_dir / baseline.latest_tree_snapshot_path
    assert v1.load_morpion_sharded_training_tree_snapshot(old_export).nodes
    retry = PublicationRunner()
    retry.grow_delta = 1
    state = run_publication(args, retry)
    assert retry.apply_calls == 1 and state.generation == 2
    assert state.latest_runtime_checkpoint_path is not None
    assert json.loads(
        (paths.work_dir / state.latest_runtime_checkpoint_path).read_bytes()
    ) == {**PATCHED, "new": 0.0}
    assert state.latest_tree_snapshot_path is not None
    exported = paths.work_dir / state.latest_tree_snapshot_path
    created = json.loads(
        (exported.parent / "node_shards/generation_000002.json").read_bytes()
    )["nodes"]
    assert [node["node_id"] for node in created] == ["new"]
    assert json.loads(exported.read_bytes())["new_node_count"] == 1


def test_complete_uncommitted_capsule_is_not_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory publication alone neither commits a generation nor acknowledges its patch."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    import chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts as receipts

    original_save = receipts.save_bootstrap_run_state

    def fail_state(state: object, path: Path, *, durable: bool = False) -> None:
        if state.generation == 2:
            message = "before state commit"
            raise OSError(message)
        original_save(state, path, durable=durable)

    with monkeypatch.context() as fault:
        fault.setattr(receipts, "save_bootstrap_run_state", fail_state)
        with pytest.raises(OSError, match="before state commit"):
            run_publication(args, PublicationRunner())
    state = load_bootstrap_run_state(paths.run_state_path)
    committed_ids = {r["id"] for r in publication_records(state.metadata)}
    orphan = next(
        p
        for p in (paths.work_dir / PUBLICATION_NAMESPACE / "objects").iterdir()
        if p.name not in committed_ids
    )
    assert (orphan / CAPSULE_SEAL).is_file()
    pending_export = orphan / "tree_exports_sharded/generation_000002.json"
    with pytest.raises(GenerationPublicationError, match="Uncommitted capsule"):
        v1.load_morpion_sharded_training_tree_snapshot(pending_export)
    with pytest.raises(GenerationPublicationError, match="Uncommitted capsule"):
        ShardedTrainingGenerationView(pending_export, cache_dir=tmp_path / "cache")
    assert paths.pipeline_reevaluation_patch_path.exists()
    retry = PublicationRunner()
    run_publication(args, retry)
    assert retry.apply_calls == 1 and not orphan.exists()


@pytest.mark.parametrize(
    "part", ["creation_shard", "updates", "catalog", "index", "model", "descriptor"]
)
def test_committed_dependency_failure_prevents_ack(tmp_path: Path, part: str) -> None:
    """A checkpoint alone cannot authorize deletion when the generation is incomplete."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths, cursor=False)
    state = run_publication(args, PublicationRunner())
    record = publication_records(state.metadata)[-1]
    root = capsule_path(paths.work_dir, record["id"])
    descriptor = read_object(root / CAPSULE_SEAL)
    relative = {
        "creation_shard": "tree_exports_sharded/node_shards/generation_000001.json",
        "updates": "tree_exports_sharded/update_shards/generation_000002.json",
        "catalog": "tree_exports_sharded/manifest.json",
        "index": "tree_exports_sharded/node_index.json",
        "model": descriptor["model_pins"][0]["path"] + "/param.pt",
        "descriptor": CAPSULE_SEAL,
    }[part]
    (root / relative).write_bytes(b"partial")
    runner = PublicationRunner()
    with pytest.raises(ValueError):
        run_publication(args, runner, max_cycles=0)
    assert runner.apply_calls == 0 and paths.pipeline_reevaluation_patch_path.exists()


def test_sealed_capsules_and_legacy_pruning_are_fenced(tmp_path: Path) -> None:
    """No supported writer may truncate an immutable hard-linked payload."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    state = run_publication(args, PublicationRunner())
    assert state.latest_tree_snapshot_path is not None
    root = (paths.work_dir / state.latest_tree_snapshot_path).parent
    before = _all_bytes(root)
    with pytest.raises(GenerationPublicationError, match="immutable publication"):
        write_v1_export(PATCHED, root, 2)
    with pytest.raises(GenerationPublicationError, match="count-based pruning"):
        prune_generation_files(paths.runtime_checkpoint_dir)
    assert _all_bytes(root) == before
    with pytest.raises(ValueError, match="durable_generation_publication=True"):
        run_pipeline_growth_stage(
            args, PublicationRunner(), max_cycles=0, checkpoint_bound_receipts=True
        )


@pytest.mark.parametrize("mode", ["flat", "both"])
def test_flat_and_both_remain_complete_publications(tmp_path: Path, mode: str) -> None:
    """Both mode pins its secondary sharded export as well as the primary flat file."""
    args, paths = setup_publication_baseline(tmp_path)
    args = replace(args, training_export_mode=mode)
    publish_fixture_patch(paths)
    state = run_publication(args, PublicationRunner())
    assert state.latest_tree_snapshot_path is not None
    flat = load_training_tree_snapshot(paths.work_dir / state.latest_tree_snapshot_path)
    assert {node.node_id: node.direct_value_scalar for node in flat.nodes} == PATCHED
    record = publication_records(state.metadata)[-1]
    descriptor = validate_capsule(paths.work_dir, record)
    assert (descriptor["sharded_snapshot_path"] is not None) == (mode == "both")
    control = PublicationRunner()
    control.values = dict(PATCHED)
    control_path = tmp_path / "control_flat.json"
    control.export_training_tree_snapshot(control_path)
    assert (
        paths.work_dir / state.latest_tree_snapshot_path
    ).read_bytes() == control_path.read_bytes()
    assert flat == load_training_tree_snapshot(control_path)
    if mode == "both":
        sharded = (
            capsule_path(paths.work_dir, record["id"])
            / descriptor["sharded_snapshot_path"]
        )
        control_root = tmp_path / "control_sharded"
        write_v1_export(BASELINE, control_root, 1)
        expected = write_v1_export(PATCHED, control_root, 2)
        assert _all_bytes(sharded.parent) == _all_bytes(control_root)
        assert v1.load_morpion_sharded_training_tree_snapshot(
            sharded
        ) == v1.load_morpion_sharded_training_tree_snapshot(expected)
