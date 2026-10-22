"""Process death and fsync faults across the existing closed-capsule transaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    capsule_path,
    publication_records,
    validate_capsule,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_durable_v2_helpers import (
    publish_scientific_patch,
    scientific_args,
)
from tests.environments.morpion_patch_receipt_helpers import PATCHED

PRE_COMMIT = [
    "partial_copy",
    "partial_updates",
    "checkpoint",
    "source_validation",
    "output_validation",
    "seal",
    "file_fsync",
    "directory_fsync",
    "before_publication",
    "after_publication",
    "before_state",
    "partial_state",
]
POST_COMMIT = [
    "after_state",
    "state_directory_fsync",
    "before_discovery",
    "during_discovery",
    "after_discovery",
    "before_ack",
    "after_ack",
]


def crash_child(root: Path, boundary: str) -> None:
    """Crash child."""
    import chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts as receipts
    import chipiron.environments.morpion.bootstrap.closed_scientific_snapshot as closed
    import chipiron.environments.morpion.bootstrap.generation_publication as pub
    from chipiron.environments.morpion.bootstrap.derived import provenance
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView as View,
    )

    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    changed = False
    original = dict(
        encode=closed._encode,
        records=View.iter_record_bytes,
        save=runner.save_checkpoint,
        validate=View.validate_source_identity,
        seal=pub.atomic_json,
        fsync=os.fsync,
        publish=pub.publish_directory_if_absent,
        replace=os.replace,
        dump=json.dump,
        manifest=receipts.GrowthPatchReceiptSession.publish_manifest,
        ack=receipts.GrowthPatchReceiptSession.acknowledge_pending,
    )

    def die() -> None:
        """Die."""
        os._exit(73)

    def encode(row: Any) -> Any:
        """Encode."""
        if (
            boundary == "partial_updates"
            and isinstance(row, dict)
            and "value_contract" in row
        ):
            for p in (root / "generation_publications_v3/staging").glob(
                "*/.closed-pending-*/updates.jsonl"
            ):
                with p.open("a") as f:
                    f.write('{"node_id":')
                    f.flush()
                die()
        return original["encode"](row)

    def records(view: Any) -> Any:
        """Interrupt after copying a complete canonical record."""
        for item in original["records"](view):
            yield item
            if boundary == "partial_copy":
                die()

    def save(path: Path) -> Any:
        """Save."""
        if boundary == "checkpoint":
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{")
            die()
        return original["save"](path)

    def validate(view: Any) -> Any:
        """Validate."""
        private = "staging" in view.manifest_path.parts
        if boundary == ("output_validation" if private else "source_validation"):
            die()
        return original["validate"](view)

    def seal(path: Path, data: Any) -> Any:
        """Seal."""
        if boundary == "seal" and path.name == ".generation_publication_v3.json":
            die()
        return original["seal"](path, data)

    def fsync(fd: Any) -> Any:
        """Fsync."""
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        private = "staging" in path.parts
        if boundary == "file_fsync" and private and path.name == "records.jsonl":
            die()
        if (
            boundary == "directory_fsync"
            and private
            and path.name == "scientific_snapshot"
        ):
            die()
        if boundary == "state_directory_fsync" and changed and path == root:
            die()
        return original["fsync"](fd)

    def publish(staging: Path, final: Path, work: Any) -> Any:
        """Publish."""
        if boundary == "before_publication" and final.parent.name == "objects":
            die()
        original["publish"](staging, final, work)
        if boundary == "after_publication" and final.parent.name == "objects":
            die()

    def replace(src: Any, dst: Any) -> Any:
        """Replace."""
        nonlocal changed
        is_state = Path(dst) == root / "run_state.json"
        if is_state and boundary == "before_state":
            die()
        original["replace"](src, dst)
        changed |= is_state
        if is_state and boundary == "after_state":
            die()

    def dump(data: Any, handle: Any, **kwargs: Any) -> Any:
        """Dump."""
        path = Path(handle.name)
        if (
            boundary == "partial_state"
            and path.parent == root
            and isinstance(data, dict)
            and "cycle_index" in data
        ) or (
            boundary == "during_discovery"
            and path.parent == root / "pipeline/generation_000002"
        ):
            handle.write("{")
            handle.flush()
            die()
        return original["dump"](data, handle, **kwargs)

    def manifest(session: Any, state: Any) -> Any:
        """Manifest."""
        if state.generation == 2 and boundary == "before_discovery":
            die()
        original["manifest"](session, state)
        if state.generation == 2 and boundary == "after_discovery":
            die()

    def ack(session: Any, state: Any) -> Any:
        """Ack."""
        if state.generation == 2 and boundary == "before_ack":
            die()
        result = original["ack"](session, state)
        if state.generation == 2 and boundary == "after_ack":
            die()
        return result

    with pytest.MonkeyPatch.context() as m:
        for target, name, func in [
            (closed, "_encode", encode),
            (View, "iter_record_bytes", records),
            (runner, "save_checkpoint", save),
            (View, "validate_source_identity", validate),
            (pub, "atomic_json", seal),
            (os, "fsync", fsync),
            (pub, "publish_directory_if_absent", publish),
            (os, "replace", replace),
            (provenance.json, "dump", dump),
            (receipts.GrowthPatchReceiptSession, "publish_manifest", manifest),
            (receipts.GrowthPatchReceiptSession, "acknowledge_pending", ack),
        ]:
            m.setattr(target, name, func)
        run_closed_publication(scientific_args(root), runner)
    message = f"Boundary not reached: {boundary}"
    raise AssertionError(message)


def run_crash(root: Path, boundary: str) -> None:
    """Run crash."""
    code = "from pathlib import Path; import sys; from tests.environments.test_morpion_durable_closed_crashes import crash_child; crash_child(Path(sys.argv[1]),sys.argv[2])"
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), boundary],
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 73, result.stderr.decode()


@pytest.mark.parametrize("boundary", PRE_COMMIT + POST_COMMIT)
def test_closed_crash_recovery(tmp_path: Path, boundary: str) -> None:
    """Verify closed crash recovery."""
    args, paths, initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    old = capsule_path(tmp_path, publication_records(initial.metadata)[0]["id"], 3)
    digest = artifact_digest(old)
    run_crash(tmp_path, boundary)
    observed = load_bootstrap_run_state(paths.run_state_path)
    committed = boundary in POST_COMMIT
    assert observed.generation == (2 if committed else 1)
    if not committed:
        assert observed == initial
    runner = ClosedPublicationRunner()
    runner.grow_delta = int(not committed)
    recovered = run_closed_publication(args, runner, max_cycles=0 if committed else 1)
    assert recovered.generation == 2 and runner.apply_calls == int(not committed)
    assert json.loads(
        (tmp_path / recovered.latest_runtime_checkpoint_path).read_text()
    ) == {**PATCHED, "new": 0.0}
    assert len(receipt_ledger(recovered)) == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert artifact_digest(old) == digest
    validate_capsule(tmp_path, publication_records(recovered.metadata)[-1])


def test_restart_interrupted_twice(tmp_path: Path) -> None:
    """Verify restart interrupted twice."""
    args, paths, _ = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    run_crash(tmp_path, "after_state")
    expected = paths.run_state_path.read_bytes()
    run_crash(tmp_path, "during_discovery")
    assert paths.run_state_path.read_bytes() == expected
    recovered = run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)
    assert recovered.generation == 2 and len(receipt_ledger(recovered)) == 1


@pytest.mark.parametrize(
    "target",
    [
        "records.jsonl",
        "updates.jsonl",
        "generation.json",
        ".generation_publication_v3.json",
        "checkpoint",
        "model",
        "capsule_directory",
        "state_directory",
    ],
)
def test_fsync_errors_preserve_observed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Verify fsync errors preserve observed commit."""
    from chipiron.environments.morpion.bootstrap import generation_publication_io as io

    args, paths, _initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    original_sync = os.fsync
    original_replace = os.replace
    changed = False
    injected = False

    def replace(src: Any, dst: Any) -> Any:
        """Replace."""
        nonlocal changed
        original_replace(src, dst)
        changed |= Path(dst) == paths.run_state_path

    def fsync(fd: Any) -> Any:
        """Fsync."""
        nonlocal injected
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        private = "staging" in path.parts
        match = (
            (private and path.name == target)
            or (target == "checkpoint" and private and path.parent.name == "checkpoint")
            or (target == "model" and private and path.name == "param.pt")
            or (
                target == "capsule_directory"
                and path == tmp_path / "generation_publications_v3/objects"
            )
            or (target == "state_directory" and changed and path == tmp_path)
        )
        if match and not injected:
            injected = True
            message = "injected fsync failure"
            raise OSError(message)
        return original_sync(fd)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", replace)
        m.setattr(os, "fsync", fsync)
        runner = ClosedPublicationRunner()
        runner.grow_delta = 1
        with pytest.raises(OSError, match="injected"):
            run_closed_publication(args, runner)
    assert injected
    observed = load_bootstrap_run_state(paths.run_state_path)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    state = run_closed_publication(
        args, runner, max_cycles=0 if observed.generation == 2 else 1
    )
    assert state.generation == 2 and len(receipt_ledger(state)) == 1
    io.validate_capsule(tmp_path, publication_records(state.metadata)[-1])
