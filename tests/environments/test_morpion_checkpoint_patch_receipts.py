"""Crash recovery for opt-in ordinary Growth; synthetic workspaces only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

import chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts as receipts
import chipiron.environments.morpion.bootstrap.derived.provenance as provenance
import chipiron.environments.morpion.bootstrap.pipeline.stages as stages
from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    load_pipeline_active_model,
    load_pipeline_manifest,
    run_pipeline_growth_stage,
    save_bootstrap_run_state,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.reevaluation_patch_consumer import (
    apply_pending_reevaluation_patch_to_runner,
)
from tests.environments.morpion_patch_receipt_helpers import (
    BASELINE,
    MODEL,
    PATCHED,
    ColdScalarRunner,
    fixture_patch,
    publish_fixture_patch,
    receipt_args,
    save_fixture_cursor,
    setup_receipt_baseline,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapPaths,
        MorpionBootstrapRunState,
    )


def _checkpoint_values(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> dict[str, float]:
    """Read only the authoritative pointer, never scan attempt outputs."""
    assert state.latest_runtime_checkpoint_path is not None
    path = paths.work_dir / state.latest_runtime_checkpoint_path
    if path.is_dir():
        path /= "nodes.json"
    return json.loads(path.read_bytes())


def _assert_committed(paths: MorpionBootstrapPaths) -> None:
    """Assert exact effects, counts, original source/model and checkpoint identity."""
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.generation == 2
    assert _checkpoint_values(paths, state) == PATCHED
    ledger = receipts.receipt_ledger(state)
    assert list(ledger) == ["patch-1"]
    receipt = ledger["patch-1"]
    assert receipt.tree_generation == 1
    assert receipt.evaluator_generation == 1
    assert receipt.evaluator_name == "default"
    assert receipt.model_bundle_path == MODEL
    assert receipt.model_sha256 == receipts.artifact_sha256(paths.work_dir / MODEL)
    assert receipt.committed_generation == 2
    assert receipt.checkpoint_path == state.latest_runtime_checkpoint_path
    assert receipt.checkpoint_sha256 == receipts.artifact_sha256(
        paths.work_dir / receipt.checkpoint_path
    )
    assert (receipt.rows_requested, receipt.rows_applied, receipt.blend_alpha) == (
        3,
        2,
        0.5,
    )


@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
def test_zero_growth_patch_forces_checkpoint_and_cold_restart(
    tmp_path: Path, checkpoint_format: Literal["json-zst", "sharded"]
) -> None:
    """An exhausted budget and unreachable save thresholds cannot discard a patch."""
    args, paths = setup_receipt_baseline(tmp_path)
    args = replace(args, runtime_checkpoint_format=checkpoint_format)
    patch = publish_fixture_patch(paths)
    patch_bytes = paths.pipeline_reevaluation_patch_path.read_bytes()
    cursor_bytes = paths.pipeline_reevaluation_cursor_path.read_bytes()
    baseline = (
        paths.work_dir
        / load_bootstrap_run_state(paths.run_state_path).latest_runtime_checkpoint_path
    ).read_bytes()
    runner = ColdScalarRunner()
    state = run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 1 and runner.values is None
    assert (
        receipts.receipt_ledger(state)[patch.patch_id].content_sha256
        == hashlib.sha256(patch_bytes).hexdigest()
    )
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == cursor_bytes
    assert runner.load_paths[0].read_bytes() == baseline
    assert not paths.pipeline_reevaluation_patch_path.exists()
    _assert_committed(paths)
    restarted = ColdScalarRunner()
    recovered = run_pipeline_growth_stage(
        args, restarted, max_cycles=2, checkpoint_bound_receipts=True
    )
    assert recovered.generation == 2 and restarted.apply_calls == 0
    assert restarted.load_paths == [
        paths.work_dir / state.latest_runtime_checkpoint_path
    ]
    _assert_committed(paths)


def test_committed_patch_waits_for_cursor_without_reapplying(tmp_path: Path) -> None:
    """A consumer cannot delete between producer publication and cursor advancement."""
    args, paths = setup_receipt_baseline(tmp_path)
    patch = publish_fixture_patch(paths, cursor=False)
    first = ColdScalarRunner()
    run_pipeline_growth_stage(args, first, checkpoint_bound_receipts=True)
    assert first.apply_calls == 1 and paths.pipeline_reevaluation_patch_path.exists()
    second = ColdScalarRunner()
    run_pipeline_growth_stage(args, second, checkpoint_bound_receipts=True)
    assert second.apply_calls == 0 and paths.pipeline_reevaluation_patch_path.exists()
    save_fixture_cursor(paths, patch)
    recovered = ColdScalarRunner()
    run_pipeline_growth_stage(
        args, recovered, max_cycles=0, checkpoint_bound_receipts=True
    )
    assert (
        recovered.apply_calls == 0
        and not paths.pipeline_reevaluation_patch_path.exists()
    )
    _assert_committed(paths)


@pytest.mark.parametrize(
    "refusal", ["checkpoint_save", "checkpoint_load", "tree_growth", "diagnostic"]
)
def test_refusal_retains_patch_and_previous_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    """RAM and diagnostic no-save paths never record a provisional application."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    if refusal == "diagnostic":
        args = replace(args, diagnostic_stop_after_growth=True)
    else:
        monkeypatch.setattr(
            stages, "log_available_ram_guard", lambda **kw: kw["action"] != refusal
        )
    runner = ColdScalarRunner()
    state = run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert state.generation == 1 and _checkpoint_values(paths, state) == BASELINE
    assert receipts.receipt_ledger(state) == {}
    assert paths.pipeline_reevaluation_patch_path.exists() and runner.values is None
    assert runner.apply_calls == int(refusal in {"checkpoint_save", "diagnostic"})
    monkeypatch.undo()
    args = replace(args, diagnostic_stop_after_growth=False)
    retry = ColdScalarRunner()
    run_pipeline_growth_stage(args, retry, checkpoint_bound_receipts=True)
    assert retry.apply_calls == 1
    _assert_committed(paths)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_state",
        "corrupt_state",
        "missing_checkpoint",
        "changed_checkpoint",
        "head_receipt",
        "schema",
    ],
)
def test_recovery_fails_closed_on_missing_or_conflicting_evidence(
    tmp_path: Path, mutation: str
) -> None:
    """No fresh root or older checkpoint fallback may inherit newer receipts."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths, cursor=False)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.latest_runtime_checkpoint_path is not None
    if mutation == "missing_state":
        paths.run_state_path.unlink()
    elif mutation == "corrupt_state":
        paths.run_state_path.write_text('{"generation":')
    elif mutation == "missing_checkpoint":
        (paths.work_dir / state.latest_runtime_checkpoint_path).unlink()
    elif mutation == "changed_checkpoint":
        (paths.work_dir / state.latest_runtime_checkpoint_path).write_text(
            '{"root": 999}'
        )
    else:
        data = state.metadata[receipts.RECEIPTS_METADATA_KEY]
        if mutation == "head_receipt":
            data["receipts"][0]["checkpoint_sha256"] = "0" * 64
        else:
            data["schema_version"] = 99
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    runner = ColdScalarRunner()
    with pytest.raises((ValueError, FileNotFoundError)):
        run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 0 and paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "future_source",
        "missing_source",
        "evaluator_generation",
        "evaluator_name",
        "model_path",
    ],
)
def test_patch_applicability_failure_preserves_patch(
    tmp_path: Path, mutation: str
) -> None:
    """Source bounds and selected evaluator identity are checked before applying."""
    args, paths = setup_receipt_baseline(tmp_path)
    changes = {
        "future_source": {"tree_generation": 2},
        "missing_source": {"tree_generation": None},
        "evaluator_generation": {"evaluator_generation": 2},
        "evaluator_name": {"evaluator_name": "other"},
        "model_path": {"model_bundle_path": "models/other"},
    }
    publish_fixture_patch(paths, replace(fixture_patch(), **changes[mutation]))
    runner = ColdScalarRunner()
    with pytest.raises(receipts.CheckpointPatchReceiptError):
        run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 0 and paths.pipeline_reevaluation_patch_path.exists()
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1


@pytest.mark.parametrize("mutation", ["bundle", "pointer"])
def test_active_model_change_before_commit_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Do not bind checkpoint effects to a different model or mutable bundle."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    runner = ColdScalarRunner()
    original_save = runner.save_checkpoint

    def replace_model(path: str | Path) -> None:
        original_save(path)
        if mutation == "bundle":
            (paths.work_dir / MODEL / "param.pt").write_bytes(b"changed-model")
        else:
            active = load_pipeline_active_model(paths.pipeline_active_model_path)
            save_pipeline_active_model(
                replace(active, generation=2), paths.pipeline_active_model_path
            )

    monkeypatch.setattr(runner, "save_checkpoint", replace_model)
    with pytest.raises(receipts.CheckpointPatchReceiptError, match="model changed"):
        run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 1 and runner.values is None
    assert paths.pipeline_reevaluation_patch_path.exists()
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1


def test_duplicate_content_is_deduplicated_and_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    """Cumulative ID/hash receipts reject changed content and survive reappearance."""
    args, paths = setup_receipt_baseline(tmp_path)
    patch = publish_fixture_patch(paths)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    publish_fixture_patch(paths, patch)
    duplicate = ColdScalarRunner()
    run_pipeline_growth_stage(args, duplicate, checkpoint_bound_receipts=True)
    assert (
        duplicate.apply_calls == 0
        and not paths.pipeline_reevaluation_patch_path.exists()
    )
    changed = replace(patch, metadata={"changed": True})
    publish_fixture_patch(paths, changed)
    conflicting = ColdScalarRunner()
    with pytest.raises(
        receipts.CheckpointPatchReceiptError, match="Conflicting reused patch ID"
    ):
        run_pipeline_growth_stage(args, conflicting, checkpoint_bound_receipts=True)
    assert (
        conflicting.apply_calls == 0 and paths.pipeline_reevaluation_patch_path.exists()
    )
    _assert_committed(paths)


def test_old_receipt_cannot_delete_new_patch(tmp_path: Path) -> None:
    """Startup acknowledgement inspects the current mailbox rather than deleting by path."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    new_patch = publish_fixture_patch(paths, fixture_patch("patch-2"))
    contents = paths.pipeline_reevaluation_patch_path.read_bytes()
    runner = ColdScalarRunner()
    run_pipeline_growth_stage(
        args, runner, max_cycles=0, checkpoint_bound_receipts=True
    )
    assert runner.apply_calls == 0
    assert (
        receipts.observe_pending_patch(paths.pipeline_reevaluation_patch_path).patch
        == new_patch
    )
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == contents


def test_flat_export_is_private_and_matches_checkpoint(tmp_path: Path) -> None:
    """The discovery projection follows commitment and keeps exact checkpoint provenance."""
    args, paths = setup_receipt_baseline(tmp_path)
    args = replace(args, growth_skip_training_export=False)
    publish_fixture_patch(paths)
    state = run_pipeline_growth_stage(
        args, ColdScalarRunner(), checkpoint_bound_receipts=True
    )
    assert state.latest_tree_snapshot_path is not None
    snapshot = paths.work_dir / state.latest_tree_snapshot_path
    assert json.loads(snapshot.read_bytes()) == PATCHED
    assert "receipt_attempts" in snapshot.parts
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))
    assert manifest.runtime_checkpoint_path == state.latest_runtime_checkpoint_path
    assert manifest.tree_snapshot_path == state.latest_tree_snapshot_path
    assert (
        manifest.metadata[receipts.RECEIPTS_METADATA_KEY]
        == state.metadata[receipts.RECEIPTS_METADATA_KEY]
    )


def test_activation_and_downgrade_guards(tmp_path: Path) -> None:
    """Opt-in does not activate unsafe exports or permit eager legacy acknowledgement."""
    args, paths = setup_receipt_baseline(tmp_path)
    args = replace(args, growth_skip_training_export=False)
    args = replace(args, training_export_mode="sharded")
    with pytest.raises(
        receipts.CheckpointPatchReceiptError, match="Activation blocked"
    ):
        run_pipeline_growth_stage(
            args, ColdScalarRunner(), max_cycles=0, checkpoint_bound_receipts=True
        )
    with pytest.raises(receipts.CheckpointPatchReceiptError, match="downgrade refused"):
        run_pipeline_growth_stage(args, ColdScalarRunner())
    publish_fixture_patch(paths)
    with pytest.raises(RuntimeError, match="identity-checked acknowledgement"):
        apply_pending_reevaluation_patch_to_runner(
            paths=paths, runner=ColdScalarRunner()
        )
    with pytest.raises(ValueError, match="atomic durable commit"):
        save_bootstrap_run_state(
            load_bootstrap_run_state(paths.run_state_path), paths.run_state_path
        )


def test_all_fsync_failures_preserve_recoverable_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail each sync in a complete transaction, including post-unlink uncertainty."""
    args, paths = setup_receipt_baseline(tmp_path / "trace")
    publish_fixture_patch(paths)
    actual_fsync = os.fsync
    sync_count = 0

    def count_sync(descriptor: int) -> None:
        nonlocal sync_count
        sync_count += 1
        actual_fsync(descriptor)

    with monkeypatch.context() as trace:
        trace.setattr(os, "fsync", count_sync)
        run_pipeline_growth_stage(
            args, ColdScalarRunner(), checkpoint_bound_receipts=True
        )
    assert sync_count > 10
    for fail_at in range(1, sync_count + 1):
        args, paths = setup_receipt_baseline(tmp_path / str(fail_at))
        publish_fixture_patch(paths)
        cursor_before = paths.pipeline_reevaluation_cursor_path.read_bytes()
        calls = 0

        def fail_sync(descriptor: int, failure_position: int = fail_at) -> None:
            nonlocal calls
            calls += 1
            if calls == failure_position:
                message = "injected fsync failure"
                raise OSError(message)
            actual_fsync(descriptor)

        with monkeypatch.context() as fault:
            fault.setattr(os, "fsync", fail_sync)
            with pytest.raises(OSError, match="injected fsync"):
                run_pipeline_growth_stage(
                    args, ColdScalarRunner(), checkpoint_bound_receipts=True
                )
        observed = load_bootstrap_run_state(paths.run_state_path)
        assert _checkpoint_values(paths, observed) == (
            PATCHED if observed.generation == 2 else BASELINE
        )
        restarted = ColdScalarRunner()
        run_pipeline_growth_stage(args, restarted, checkpoint_bound_receipts=True)
        assert restarted.apply_calls == int(observed.generation == 1)
        assert paths.pipeline_reevaluation_cursor_path.read_bytes() == cursor_before
        _assert_committed(paths)


def test_partial_application_never_commits_and_same_runner_cold_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discard a partly mutated runtime even if the caller reuses its adapter."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    runner = ColdScalarRunner()
    original_apply = runner.apply_reevaluation_patch

    def partial_apply(_patch: object) -> int:
        assert runner.values is not None
        runner.values["root"] = 6.0
        message = "injected partial application"
        raise RuntimeError(message)

    with monkeypatch.context() as failure:
        failure.setattr(runner, "apply_reevaluation_patch", partial_apply)
        with pytest.raises(RuntimeError, match="partial application"):
            run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.values is None and paths.pipeline_reevaluation_patch_path.exists()
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1
    assert runner.apply_reevaluation_patch == original_apply
    run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 1
    _assert_committed(paths)


def test_partial_run_state_temporary_is_ignored_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted JSON serialization cannot damage the previous committed JSON."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    before = paths.run_state_path.read_bytes()
    original_dump = json.dump

    def partial_dump(data: object, handle: object, **kwargs: object) -> None:
        if isinstance(data, dict) and data.get("generation") == 2:
            handle.write('{"generation":')
            handle.flush()
            message = "injected temporary write failure"
            raise OSError(message)
        original_dump(data, handle, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(provenance.json, "dump", partial_dump)
        with pytest.raises(OSError, match="temporary write failure"):
            run_pipeline_growth_stage(
                args, ColdScalarRunner(), checkpoint_bound_receipts=True
            )
    assert paths.run_state_path.read_bytes() == before
    assert paths.pipeline_reevaluation_patch_path.exists()
    orphaned = [p for p in tmp_path.glob("tmp*") if p.is_file()]
    assert orphaned and any(p.read_text() == '{"generation":' for p in orphaned)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    _assert_committed(paths)
    assert all(p.exists() for p in orphaned)


def test_atomic_run_state_preserves_legacy_bytes(tmp_path: Path) -> None:
    """Durable publication reuses the exact existing serializer and trailing newline."""
    _args, paths = setup_receipt_baseline(tmp_path)
    state = load_bootstrap_run_state(paths.run_state_path)
    legacy = replace(state, metadata={"scientific": [1, 2.0, {"unicode": "é"}]})
    ordinary, durable = tmp_path / "ordinary.json", tmp_path / "durable.json"
    save_bootstrap_run_state(legacy, ordinary)
    save_bootstrap_run_state(legacy, durable, durable=True)
    assert ordinary.read_bytes() == durable.read_bytes()


def test_receipts_survive_later_generations_and_old_patch_reappearance(
    tmp_path: Path,
) -> None:
    """A receipt remains attached to descendant checkpoints and is never replaced by last ID."""
    args, paths = setup_receipt_baseline(tmp_path)
    original = publish_fixture_patch(paths)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    publish_fixture_patch(paths, fixture_patch("patch-2"))
    second = run_pipeline_growth_stage(
        args, ColdScalarRunner(), checkpoint_bound_receipts=True
    )
    assert second.generation == 3
    assert _checkpoint_values(paths, second) == {"root": 8.0, "leaf": 10.0}
    assert set(receipts.receipt_ledger(second)) == {"patch-1", "patch-2"}
    publish_fixture_patch(paths, original)
    recovered = ColdScalarRunner()
    state = run_pipeline_growth_stage(args, recovered, checkpoint_bound_receipts=True)
    assert recovered.apply_calls == 0 and state.generation == 3
    assert _checkpoint_values(paths, state) == {"root": 8.0, "leaf": 10.0}


@pytest.mark.parametrize("corrupt_bundle", [False, True])
def test_committed_ack_uses_original_model_identity(
    tmp_path: Path, corrupt_bundle: bool
) -> None:
    """A later active pointer cannot change a receipt's original model provenance."""
    args, paths = setup_receipt_baseline(tmp_path)
    patch = publish_fixture_patch(paths, cursor=False)
    run_pipeline_growth_stage(args, ColdScalarRunner(), checkpoint_bound_receipts=True)
    active = load_pipeline_active_model(paths.pipeline_active_model_path)
    save_pipeline_active_model(
        replace(active, generation=2), paths.pipeline_active_model_path
    )
    if corrupt_bundle:
        (tmp_path / MODEL / "param.pt").write_bytes(b"different-original-bundle")
    save_fixture_cursor(paths, patch)
    runner = ColdScalarRunner()
    if corrupt_bundle:
        with pytest.raises(
            receipts.CheckpointPatchReceiptError, match="model content changed"
        ):
            run_pipeline_growth_stage(
                args, runner, max_cycles=0, checkpoint_bound_receipts=True
            )
        assert paths.pipeline_reevaluation_patch_path.exists()
    else:
        run_pipeline_growth_stage(
            args, runner, max_cycles=0, checkpoint_bound_receipts=True
        )
        assert not paths.pipeline_reevaluation_patch_path.exists()
        _assert_committed(paths)
    assert runner.apply_calls == 0


@pytest.mark.parametrize(
    "operation", ["before_replace", "after_replace", "before_unlink", "after_unlink"]
)
def test_publication_errors_inspect_actual_state_without_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """An exception after a namespace operation never implies it did not happen."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    original_replace, original_unlink = os.replace, Path.unlink

    def fail() -> None:
        message = "injected namespace failure"
        raise OSError(message)

    def replace_state(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == paths.run_state_path and operation == "before_replace":
            fail()
        original_replace(source, destination)
        if Path(destination) == paths.run_state_path and operation == "after_replace":
            fail()

    def unlink_patch(path: Path, missing_ok: bool = False) -> None:
        if (
            path == paths.pipeline_reevaluation_patch_path
            and operation == "before_unlink"
        ):
            fail()
        original_unlink(path, missing_ok=missing_ok)
        if (
            path == paths.pipeline_reevaluation_patch_path
            and operation == "after_unlink"
        ):
            fail()

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", replace_state)
        fault.setattr(Path, "unlink", unlink_patch)
        with pytest.raises(OSError, match="namespace failure"):
            run_pipeline_growth_stage(
                args, ColdScalarRunner(), checkpoint_bound_receipts=True
            )
    observed = load_bootstrap_run_state(paths.run_state_path)
    committed = operation != "before_replace"
    assert observed.generation == (2 if committed else 1)
    assert paths.pipeline_reevaluation_patch_path.exists() == (
        operation != "after_unlink"
    )
    runner = ColdScalarRunner()
    run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == int(not committed)
    _assert_committed(paths)


@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
def test_filesystem_commit_and_ack_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_format: Literal["json-zst", "sharded"],
) -> None:
    """Every dependency precedes the commit, then cursor durability precedes unlink."""
    args, paths = setup_receipt_baseline(tmp_path)
    args = replace(
        args,
        runtime_checkpoint_format=checkpoint_format,
        growth_skip_training_export=False,
    )
    publish_fixture_patch(paths)
    events: list[tuple[str, str]] = []
    original_sync, original_replace, original_unlink = os.fsync, os.replace, Path.unlink

    def sync(descriptor: int) -> None:
        original_sync(descriptor)
        events.append(("sync", os.readlink(f"/proc/self/fd/{descriptor}")))

    def publish(source: str | Path, destination: str | Path) -> None:
        original_replace(source, destination)
        events.append(("replace", str(destination)))

    def unlink(path: Path, missing_ok: bool = False) -> None:
        original_unlink(path, missing_ok=missing_ok)
        events.append(("unlink", str(path)))

    with monkeypatch.context() as trace:
        trace.setattr(os, "fsync", sync)
        trace.setattr(os, "replace", publish)
        trace.setattr(Path, "unlink", unlink)
        state = run_pipeline_growth_stage(
            args, ColdScalarRunner(), checkpoint_bound_receipts=True
        )
    state_replace = events.index(("replace", str(paths.run_state_path)))
    state_commit = events.index(("sync", str(tmp_path)), state_replace)
    unlink_index = events.index(("unlink", str(paths.pipeline_reevaluation_patch_path)))
    cursor_sync = events.index(("sync", str(paths.pipeline_reevaluation_cursor_path)))
    assert state_replace < state_commit < cursor_sync < unlink_index
    assert ("sync", str(paths.pipeline_dir)) in events[unlink_index + 1 :]
    assert state.latest_runtime_checkpoint_path is not None
    assert state.latest_tree_snapshot_path is not None
    checkpoint = tmp_path / state.latest_runtime_checkpoint_path
    dependencies = (
        [p for p in checkpoint.rglob("*") if p.is_file()]
        if checkpoint.is_dir()
        else [checkpoint]
    )
    dependencies += [
        tmp_path / state.latest_tree_snapshot_path,
        tmp_path / MODEL / "param.pt",
    ]
    for path in dependencies:
        assert ("sync", str(path)) in events[:state_replace]
        assert ("sync", str(path.parent)) in events[:state_replace]
    manifest_replace = events.index((
        "replace",
        str(paths.pipeline_manifest_path_for_generation(2)),
    ))
    assert state_commit < manifest_replace < unlink_index


def test_activation_refuses_legacy_pending_patch(tmp_path: Path) -> None:
    """A legacy pending patch has no proof that its in-memory effects were uncommitted."""
    args, paths = setup_receipt_baseline(tmp_path, adopt=False)
    publish_fixture_patch(paths)
    before = paths.run_state_path.read_bytes()
    with pytest.raises(
        receipts.CheckpointPatchReceiptError, match="quiescent baseline"
    ):
        run_pipeline_growth_stage(
            args, ColdScalarRunner(), checkpoint_bound_receipts=True
        )
    assert paths.pipeline_reevaluation_patch_path.exists()
    assert paths.run_state_path.read_bytes() == before


def test_partial_pending_json_fails_without_deletion(tmp_path: Path) -> None:
    """Legacy or corrupt input is never mistaken for an absent pending patch."""
    args, paths = setup_receipt_baseline(tmp_path)
    paths.pipeline_reevaluation_patch_path.write_text('{"patch_id":')
    with pytest.raises(json.JSONDecodeError):
        run_pipeline_growth_stage(
            args, ColdScalarRunner(), checkpoint_bound_receipts=True
        )
    assert paths.pipeline_reevaluation_patch_path.read_text() == '{"patch_id":'
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1


CRASH_BOUNDARIES = [
    "before_apply",
    "after_apply",
    "during_checkpoint",
    "after_checkpoint",
    "before_state_replace",
    "after_state_replace",
    "before_manifest",
    "after_manifest",
    "before_ack",
    "before_unlink",
    "after_unlink",
    "after_ack",
]


def _crash_growth(work_dir: Path, boundary: str) -> None:
    """Child-only deterministic process death; never launch a production worker."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    args = receipt_args(work_dir)
    args = replace(args, growth_skip_training_export=False)
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    runner = ColdScalarRunner()
    original_apply = runner.apply_reevaluation_patch
    original_save = runner.save_checkpoint
    original_replace = os.replace
    original_unlink = Path.unlink
    original_manifest = receipts.GrowthPatchReceiptSession.publish_manifest
    original_ack = receipts.GrowthPatchReceiptSession.acknowledge_pending

    def die() -> None:
        os._exit(71)

    def apply(patch: object) -> int:
        if boundary == "before_apply":
            die()
        result = original_apply(patch)
        if boundary == "after_apply":
            die()
        return result

    def save(path: str | Path) -> None:
        if boundary == "during_checkpoint":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{")
            die()
        original_save(path)
        if boundary == "after_checkpoint":
            die()

    def state_replace(source: str | Path, destination: str | Path) -> None:
        if (
            Path(destination) == paths.run_state_path
            and boundary == "before_state_replace"
        ):
            die()
        original_replace(source, destination)
        if (
            Path(destination) == paths.run_state_path
            and boundary == "after_state_replace"
        ):
            die()

    def manifest(
        session: receipts.GrowthPatchReceiptSession, state: MorpionBootstrapRunState
    ) -> None:
        if state.generation == 2 and boundary == "before_manifest":
            die()
        original_manifest(session, state)
        if state.generation == 2 and boundary == "after_manifest":
            die()

    def ack(
        session: receipts.GrowthPatchReceiptSession, state: MorpionBootstrapRunState
    ) -> bool:
        if state.generation == 2 and boundary == "before_ack":
            die()
        result = original_ack(session, state)
        if state.generation == 2 and boundary == "after_ack":
            die()
        return result

    def unlink(path: Path, *args: object, **kwargs: object) -> None:
        if (
            path == paths.pipeline_reevaluation_patch_path
            and boundary == "before_unlink"
        ):
            die()
        original_unlink(path, *args, **kwargs)
        if (
            path == paths.pipeline_reevaluation_patch_path
            and boundary == "after_unlink"
        ):
            die()

    with pytest.MonkeyPatch.context() as fault:
        fault.setattr(runner, "apply_reevaluation_patch", apply)
        fault.setattr(runner, "save_checkpoint", save)
        fault.setattr(os, "replace", state_replace)
        fault.setattr(Path, "unlink", unlink)
        fault.setattr(receipts.GrowthPatchReceiptSession, "publish_manifest", manifest)
        fault.setattr(receipts.GrowthPatchReceiptSession, "acknowledge_pending", ack)
        run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    message = "Crash injection was not reached"
    raise AssertionError(message)


@pytest.mark.parametrize("boundary", CRASH_BOUNDARIES)
def test_process_crash_recovery(tmp_path: Path, boundary: str) -> None:
    """SIGKILL-equivalent exit cannot run cleanup; restart uses only persisted evidence."""
    args, paths = setup_receipt_baseline(tmp_path)
    args = replace(args, growth_skip_training_export=False)
    publish_fixture_patch(paths)
    previous_bytes = paths.run_state_path.read_bytes()
    script = (
        "from pathlib import Path; from tests.environments.test_morpion_checkpoint_patch_receipts "
        "import _crash_growth; import sys; _crash_growth(Path(sys.argv[1]), sys.argv[2])"
    )
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 71, process.stderr.decode()
    state = load_bootstrap_run_state(paths.run_state_path)
    committed = boundary not in {
        "before_apply",
        "after_apply",
        "during_checkpoint",
        "after_checkpoint",
        "before_state_replace",
    }
    assert state.generation == (2 if committed else 1)
    if not committed:
        assert paths.run_state_path.read_bytes() == previous_bytes
        assert paths.pipeline_reevaluation_patch_path.exists()
        assert not paths.pipeline_manifest_path_for_generation(2).exists()
    runner = ColdScalarRunner()
    run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == int(not committed)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    _assert_committed(paths)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))
    assert (
        manifest.runtime_checkpoint_path
        == load_bootstrap_run_state(paths.run_state_path).latest_runtime_checkpoint_path
    )
