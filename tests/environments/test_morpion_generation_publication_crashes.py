"""Deterministic process death and durability errors using generated artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts as receipts
import chipiron.environments.morpion.bootstrap.derived.provenance as provenance
import chipiron.environments.morpion.bootstrap.generation_publication as publication
import chipiron.environments.morpion.bootstrap.sharded_training_export as v1
from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    ATTEMPT_MARKER,
    CAPSULE_SEAL,
    PUBLICATION_KEY,
    PUBLICATION_NAMESPACE,
    capsule_path,
    publication_records,
    read_object,
    validate_capsule,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import (
    PATCHED,
    publish_fixture_patch,
    receipt_args,
)

BOUNDARIES = [
    "before_shard",
    "during_shard",
    "after_shards",
    "before_catalog",
    "after_catalog",
    "after_index",
    "before_capsule",
    "after_capsule",
    "before_state",
    "after_state",
    "before_manifest",
    "during_manifest",
    "after_manifest",
    "before_ack",
    "before_unlink",
    "after_unlink",
    "after_ack",
    "during_prune",
    "after_prune",
]
BEFORE_COMMIT = set(BOUNDARIES[:9])


def _crash_child(work_dir: Path, boundary: str) -> None:
    """Fault one real filesystem boundary; os._exit bypasses cleanup/finalizers."""
    args = replace(
        receipt_args(work_dir),
        training_export_mode="sharded",
        growth_skip_training_export=False,
    )
    runner = PublicationRunner()
    runner.grow_delta = 1
    original_write = v1._write_json
    original_publish = publication.publish_directory_if_absent
    original_replace = os.replace
    original_dump = json.dump
    original_manifest = receipts.GrowthPatchReceiptSession.publish_manifest
    original_ack = receipts.GrowthPatchReceiptSession.acknowledge_pending
    original_unlink = Path.unlink
    original_prune = publication.DurableGenerationPublication.prune_uncommitted

    def die() -> None:
        os._exit(73)

    def write(path: Path, payload: dict[str, object]) -> object:
        if path.name == "generation_000002.json":
            if path.parent.name == "node_shards" and boundary in {
                "before_shard",
                "during_shard",
            }:
                if boundary == "during_shard":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{"nodes": [')
                die()
            if (
                path.parent.name == "tree_exports_sharded"
                and boundary == "after_shards"
            ):
                die()
        new_catalog = (
            path.name == "manifest.json" and payload.get("latest_generation") == 2
        )
        if new_catalog and boundary == "before_catalog":
            die()
        result = original_write(path, payload)
        if new_catalog and boundary == "after_catalog":
            die()
        if (
            path.name == "node_index.json"
            and boundary == "after_index"
            and "new" in payload.get("node_id_to_creation_generation", {})
        ):
            die()
        return result

    def publish(staging: Path, final: Path, root: Path) -> None:
        if final.parent.name == "objects" and boundary == "before_capsule":
            die()
        original_publish(staging, final, root)
        if final.parent.name == "objects" and boundary == "after_capsule":
            die()

    def state_replace(source: str | Path, destination: str | Path) -> None:
        is_state = Path(destination) == work_dir / "run_state.json"
        if is_state and boundary == "before_state":
            die()
        original_replace(source, destination)
        if is_state and boundary == "after_state":
            die()

    def dump(data: object, handle: object, **kwargs: object) -> None:
        if (
            boundary == "during_manifest"
            and Path(handle.name).parent == work_dir / "pipeline/generation_000002"
        ):
            handle.write('{"generation":')
            handle.flush()
            die()
        original_dump(data, handle, **kwargs)

    def manifest(session: receipts.GrowthPatchReceiptSession, state: object) -> None:
        if state.generation == 2 and boundary == "before_manifest":
            die()
        original_manifest(session, state)
        if state.generation == 2 and boundary == "after_manifest":
            die()

    def ack(session: receipts.GrowthPatchReceiptSession, state: object) -> bool:
        if state.generation == 2 and boundary == "before_ack":
            die()
        result = original_ack(session, state)
        if state.generation == 2 and boundary == "after_ack":
            die()
        return result

    def unlink(path: Path, missing_ok: bool = False) -> None:
        is_patch = path == work_dir / "pipeline/reevaluation_patch.json"
        if is_patch and boundary == "before_unlink":
            die()
        original_unlink(path, missing_ok=missing_ok)
        if (is_patch and boundary == "after_unlink") or (
            boundary == "during_prune"
            and "trash" in path.parts
            and path.name == "a_payload"
        ):
            die()

    def prune(
        context: publication.DurableGenerationPublication, state: object
    ) -> list[str]:
        if state.generation == 2 and boundary in {"during_prune", "after_prune"}:
            identifier = uuid.uuid4().hex
            path = work_dir / PUBLICATION_NAMESPACE / "staging" / identifier
            path.mkdir(parents=True)
            provenance.atomic_json(
                path / ATTEMPT_MARKER,
                {"protocol": PUBLICATION_KEY, "id": identifier, "generation": 99},
            )
            (path / "a_payload").write_bytes(b"abandoned")
            (path / "b_payload").write_bytes(b"still abandoned")
        result = original_prune(context, state)
        if state.generation == 2 and boundary == "after_prune":
            die()
        return result

    with pytest.MonkeyPatch.context() as fault:
        fault.setattr(v1, "_write_json", write)
        fault.setattr(publication, "publish_directory_if_absent", publish)
        fault.setattr(os, "replace", state_replace)
        fault.setattr(provenance.json, "dump", dump)
        fault.setattr(receipts.GrowthPatchReceiptSession, "publish_manifest", manifest)
        fault.setattr(receipts.GrowthPatchReceiptSession, "acknowledge_pending", ack)
        fault.setattr(Path, "unlink", unlink)
        fault.setattr(
            publication.DurableGenerationPublication, "prune_uncommitted", prune
        )
        run_publication(args, runner)
    message = "Crash boundary was not reached"
    raise AssertionError(message)


@pytest.mark.parametrize(
    "boundary",
    [
        pytest.param(
            value,
            marks=()
            if value in ["after_state", "before_state"]
            else pytest.mark.exhaustive,
        )
        for value in BOUNDARIES
    ],
)
def test_crash_recovery_at_publication_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    """A durable head survives; uncommitted output never changes the restore lineage."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    previous = load_bootstrap_run_state(paths.run_state_path)
    baseline_record = publication_records(previous.metadata)[0]
    baseline_root = capsule_path(paths.work_dir, baseline_record["id"])
    baseline_digest = receipts.artifact_sha256(baseline_root)
    script = (
        "from pathlib import Path; import sys; "
        "from tests.environments.test_morpion_generation_publication_crashes import _crash_child; "
        "_crash_child(Path(sys.argv[1]),sys.argv[2])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 73, result.stderr.decode()
    observed = load_bootstrap_run_state(paths.run_state_path)
    committed = boundary not in BEFORE_COMMIT
    assert observed.generation == (2 if committed else 1)
    if not committed:
        assert observed == previous
        assert paths.pipeline_reevaluation_patch_path.exists()
    recovered = PublicationRunner()
    recovered.grow_delta = int(not committed)
    state = run_publication(args, recovered)
    assert state.generation == 2 and recovered.apply_calls == int(not committed)
    assert state.latest_runtime_checkpoint_path is not None
    assert json.loads(
        (paths.work_dir / state.latest_runtime_checkpoint_path).read_bytes()
    ) == {**PATCHED, "new": 0.0}
    assert receipts.artifact_sha256(baseline_root) == baseline_digest
    assert len(publication_records(state.metadata)) == 2
    assert len(receipts.receipt_ledger(state)) == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()
    record = publication_records(state.metadata)[-1]
    validate_capsule(paths.work_dir, record)
    assert not list((paths.work_dir / PUBLICATION_NAMESPACE / "trash").iterdir())


def test_crash_during_manifest_recovery_preserves_committed_receipt(
    tmp_path: Path,
) -> None:
    """A second crash while repairing discovery never rolls back committed effects."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    script = (
        "from pathlib import Path; import sys; "
        "from tests.environments.test_morpion_generation_publication_crashes import _crash_child; "
        "_crash_child(Path(sys.argv[1]),sys.argv[2])"
    )
    committed = None
    for boundary in ["after_state", "during_manifest"]:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), boundary],
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 73, result.stderr.decode()
        observed = load_bootstrap_run_state(paths.run_state_path)
        assert observed.generation == 2
        if committed is not None:
            assert observed == committed
        committed = observed
        assert paths.pipeline_reevaluation_patch_path.exists()
        assert not paths.pipeline_manifest_path_for_generation(2).exists()
    runner = PublicationRunner()
    recovered = run_publication(args, runner, max_cycles=0)
    assert recovered == committed and runner.apply_calls == 0
    assert not paths.pipeline_reevaluation_patch_path.exists()
    manifest = read_object(paths.pipeline_manifest_path_for_generation(2))
    assert (
        manifest["runtime_checkpoint_path"] == recovered.latest_runtime_checkpoint_path
    )
    assert manifest["tree_snapshot_path"] == recovered.latest_tree_snapshot_path
    validate_capsule(tmp_path, publication_records(recovered.metadata)[-1])


@pytest.mark.parametrize(
    "target",
    [
        "creation",
        "updates",
        "catalog",
        "index",
        "seal",
        "model",
        "checkpoint",
        "capsule_directory",
        "state_directory",
        "ack_directory",
    ],
)
def test_fsync_failure_preserves_recoverable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """I/O failure after visibility is resolved from observed state, never from the exception."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    original_sync = os.fsync
    original_replace = os.replace
    original_unlink = Path.unlink
    state_replaced = False
    unlinked = False
    injected = False

    def publish(source: str | Path, destination: str | Path) -> None:
        nonlocal state_replaced
        original_replace(source, destination)
        state_replaced |= Path(destination) == paths.run_state_path

    def unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal unlinked
        original_unlink(path, missing_ok=missing_ok)
        unlinked |= path == paths.pipeline_reevaluation_patch_path

    def fsync(descriptor: int) -> None:
        nonlocal injected
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        in_attempt = "staging" in path.parts and PUBLICATION_NAMESPACE in path.parts
        match = {
            "creation": in_attempt
            and path.name == "generation_000002.json"
            and path.parent.name == "node_shards",
            "updates": in_attempt
            and path.name == "generation_000002.json"
            and path.parent.name == "update_shards",
            "catalog": in_attempt
            and path.name == "manifest.json"
            and path.parent.name == "tree_exports_sharded",
            "index": in_attempt and path.name == "node_index.json",
            "seal": in_attempt and path.name == CAPSULE_SEAL,
            "model": in_attempt and path.name == "param.pt",
            "checkpoint": in_attempt and path.name == "generation_000002.json.zst",
            "capsule_directory": path == tmp_path / PUBLICATION_NAMESPACE / "objects",
            "state_directory": state_replaced and path == tmp_path,
            "ack_directory": unlinked and path == paths.pipeline_dir,
        }[target]
        if match and not injected:
            injected = True
            message = "injected publication fsync failure"
            raise OSError(message)
        original_sync(descriptor)

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fsync)
        fault.setattr(os, "replace", publish)
        fault.setattr(Path, "unlink", unlink)
        with pytest.raises(OSError, match="publication fsync failure"):
            run_publication(args, PublicationRunner())
    assert injected
    observed = load_bootstrap_run_state(paths.run_state_path)
    runner = PublicationRunner()
    state = run_publication(args, runner)
    assert runner.apply_calls == int(observed.generation == 1)
    assert state.generation == 2
    assert state.latest_runtime_checkpoint_path is not None
    assert (
        json.loads((tmp_path / state.latest_runtime_checkpoint_path).read_bytes())
        == PATCHED
    )
    assert not paths.pipeline_reevaluation_patch_path.exists()
