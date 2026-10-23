"""Executable counterexamples, NOT an implementation of research epochs.

All workspaces are new pytest fixtures. Passing a characterization of an unsafe
shortcut is evidence to keep that shortcut disabled, not a safety approval.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    load_pipeline_active_model,
    load_pipeline_manifest,
    save_bootstrap_run_state,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    RECEIPTS_METADATA_KEY,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_scientific_source,
    generation_read_pin,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    publish_closed_patch,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_durable_v2_helpers import scientific_producer
from tests.environments.morpion_patch_receipt_helpers import (
    PATCHED,
    fixture_patch,
    publish_fixture_patch,
    save_fixture_cursor,
)


def sourced_patch(paths: Any, identity: str = "patch-before-crash") -> Any:
    """Use the actual committed scientific source; no fabricated capsule identity."""
    state = load_bootstrap_run_state(paths.run_state_path)
    return replace(
        fixture_patch(identity),
        tree_generation=state.generation,
        metadata={
            "scientific_publication": committed_scientific_source(
                paths.work_dir / state.latest_tree_snapshot_path
            ),
        },
    )


def test_patch_commit_without_cursor_is_safe_but_cannot_finish_recovery(
    tmp_path: Path,
) -> None:
    """Real commit deduplicates, but three cold restarts cannot reconstruct the cursor."""
    args, paths, _ = setup_closed_baseline(tmp_path)
    patch = sourced_patch(paths)
    publish_fixture_patch(paths, patch, cursor=False)
    first = ClosedPublicationRunner()
    state = run_closed_publication(args, first)
    assert first.apply_calls == 1
    assert (
        json.loads((tmp_path / state.latest_runtime_checkpoint_path).read_bytes())
        == PATCHED
    )
    assert len(receipt_ledger(state)) == 1
    for _ in range(3):
        restarted = ClosedPublicationRunner()
        state = run_closed_publication(args, restarted, max_cycles=0)
        assert restarted.apply_calls == 0
        assert paths.pipeline_reevaluation_patch_path.exists()
        assert not paths.pipeline_reevaluation_cursor_path.exists()
        assert len(receipt_ledger(state)) == 1
    # Explicitly supplied missing producer information resolves the stall;
    # this is a diagnostic control, NOT a proposed automatic repair.
    save_fixture_cursor(paths, patch)
    run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)
    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_epoch_label_alone_does_not_fence_patch(tmp_path: Path) -> None:
    """Changing metadata in place cannot safely implement a fork."""
    args, paths, state = setup_closed_baseline(tmp_path)
    patch = sourced_patch(paths)
    patch = replace(patch, metadata={**patch.metadata, "epoch_id": "old-epoch"})
    state = replace(state, metadata={**state.metadata, "epoch_id": "new-epoch"})
    save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    publish_fixture_patch(paths, patch)
    runner = ClosedPublicationRunner()
    committed = run_closed_publication(args, runner)
    assert runner.apply_calls == 1  # Demonstrated missing epoch guard.
    assert committed.metadata["epoch_id"] == "new-epoch"
    assert patch.patch_id in receipt_ledger(committed)


def test_fresh_capsule_identity_rejects_another_workspace_patch(tmp_path: Path) -> None:
    """Existing source validation helps; it is not a cross-stage epoch protocol."""
    _, old_paths, _ = setup_closed_baseline(tmp_path / "old")
    args, new_paths, _ = setup_closed_baseline(tmp_path / "new")
    publish_fixture_patch(new_paths, sourced_patch(old_paths))
    runner = ClosedPublicationRunner()
    with pytest.raises(ValueError, match="scientific publication identity mismatch"):
        run_closed_publication(args, runner)
    assert runner.apply_calls == 0
    assert not receipt_ledger(load_bootstrap_run_state(new_paths.run_state_path))


@pytest.mark.parametrize("erase_receipts", [False, True])
def test_retagging_does_not_make_receipt_erasure_safe(
    tmp_path: Path, erase_receipts: bool
) -> None:
    """Clearing receipts after a cosmetic epoch change reapplies real scalar effects."""
    args, paths, _ = setup_closed_baseline(tmp_path)
    patch = sourced_patch(paths)
    publish_fixture_patch(paths, patch)
    committed = run_closed_publication(args, ClosedPublicationRunner())
    metadata = {**committed.metadata, "epoch_id": "cosmetic-new-epoch"}
    if erase_receipts:
        metadata[RECEIPTS_METADATA_KEY] = {
            **metadata[RECEIPTS_METADATA_KEY],
            "receipts": [],
        }
    save_bootstrap_run_state(
        replace(committed, metadata=metadata), paths.run_state_path, durable=True
    )
    publish_fixture_patch(paths, patch)
    runner = ClosedPublicationRunner()
    retried = run_closed_publication(args, runner)
    assert runner.apply_calls == int(erase_receipts)
    values = json.loads(
        (tmp_path / retried.latest_runtime_checkpoint_path).read_bytes()
    )
    assert values == ({"root": 8.0, "leaf": 10.0} if erase_receipts else PATCHED)
    assert len(receipt_ledger(retried)) == 1


def test_retained_non_genesis_snapshot_is_not_an_existing_fork_api(
    tmp_path: Path,
) -> None:
    """The initializer refuses G2 before attempting to adopt a different workspace."""
    args, paths, _ = setup_closed_baseline(tmp_path / "source")
    publish_closed_patch(paths, "receipt-2")
    older = run_closed_publication(args, ClosedPublicationRunner())
    publish_closed_patch(paths, "receipt-3")
    run_closed_publication(args, ClosedPublicationRunner())
    from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths

    target = MorpionBootstrapPaths.from_work_dir(tmp_path / "target")
    target.ensure_directories()
    with (
        generation_read_pin(paths.work_dir / older.latest_tree_snapshot_path),
        pytest.raises(ValueError, match="must be generation 1"),
    ):
        initialize_scientific_v2_publication(
            target,
            older,
            producer=scientific_producer(args),
            closed_scientific_snapshot=True,
        )
    assert not target.run_state_path.exists()
    assert load_bootstrap_run_state(paths.run_state_path).generation == 3


def test_epoch_label_alone_does_not_fence_late_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual Training completion path ignores a changed epoch label.

    This intentionally invalid in-place retag is the shortcut being disproved;
    no real new-epoch fork is claimed. The source remains a validated v3 capsule.
    """
    from tests.environments.test_morpion_bootstrap_pipeline_stages import (
        _artifact_pipeline_args,
        _fake_training_result,
        _prepare_training_stage_input,
    )

    args, paths, _ = setup_closed_baseline(tmp_path)
    publish_closed_patch(paths, "make-g2")
    run_closed_publication(args, ClosedPublicationRunner())
    manifest_path = paths.pipeline_manifest_path_for_generation(2)
    sealed_manifest = load_pipeline_manifest(manifest_path)
    _prepare_training_stage_input(paths, generation=2)
    ready = load_pipeline_manifest(manifest_path)
    save_pipeline_manifest(
        replace(
            sealed_manifest,
            rows_path=ready.rows_path,
            dataset_status="done",
            metadata={**sealed_manifest.metadata, "epoch_id": "old-epoch"},
        ),
        manifest_path,
    )

    def finish_old_training(**_kwargs: Any) -> Any:
        latest = load_bootstrap_run_state(paths.run_state_path)
        save_bootstrap_run_state(
            replace(latest, metadata={**latest.metadata, "epoch_id": "new-epoch"}),
            paths.run_state_path,
            durable=True,
        )
        return _fake_training_result(paths, generation=2)

    monkeypatch.setattr(stages, "_train_and_select_evaluators", finish_old_training)
    done = stages.run_pipeline_training_stage(
        _artifact_pipeline_args(tmp_path), generation=2
    )
    assert done.training_status == "done"
    assert done.metadata["epoch_id"] == "old-epoch"
    assert (
        load_bootstrap_run_state(paths.run_state_path).metadata["epoch_id"]
        == "new-epoch"
    )
    assert load_pipeline_active_model(paths.pipeline_active_model_path).generation == 2


def test_growth_lock_and_old_reader_pin_coexist(tmp_path: Path) -> None:
    """Existing kernel locks, unlike claim TTLs, are actual ownership evidence."""
    args, paths, original = setup_closed_baseline(tmp_path)
    with generation_read_pin(tmp_path / original.latest_tree_snapshot_path):
        publish_closed_patch(paths, "new-generation")
        state = run_closed_publication(args, ClosedPublicationRunner())
        assert state.generation == 2
        assert len(publication_records(state.metadata)) == 2
        with (
            growth_receipt_ownership(paths),
            pytest.raises((BlockingIOError, ValueError)),
            growth_receipt_ownership(paths),
        ):
            pytest.fail("competing Growth entered")
