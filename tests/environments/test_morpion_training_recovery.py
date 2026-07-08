"""Tests for stale Morpion pipeline training recovery."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline.training_recovery import (
    inspect_training_state_for_recovery,
    main,
    recover_stale_training_state,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
    save_pipeline_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_NOW = datetime(2026, 7, 8, 16, 0, tzinfo=UTC)


def test_expired_claim_empty_training_status_manifest_training_is_stale(
    tmp_path: Path,
) -> None:
    """Expired claim plus empty in-progress status should be recoverable."""
    generation_dir = _stale_generation_fixture(tmp_path)

    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    assert state.is_stale
    assert state.reason == "expired_training_claim_empty_evaluator_results"
    assert state.claim_expired is True
    assert state.status_file_status == "training"
    assert state.manifest_training_status == "training"
    assert state.evaluator_results_count == 0


def test_non_expired_claim_is_not_stale(tmp_path: Path) -> None:
    """A live claim must remain protected from auto-recovery."""
    generation_dir = _stale_generation_fixture(
        tmp_path,
        expires_at_utc="2026-07-08T16:30:00Z",
    )

    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    assert not state.is_stale
    assert state.reason == "active_training_claim"


def test_done_status_with_evaluator_results_is_not_stale(tmp_path: Path) -> None:
    """Completed evaluator results must never be reclaimed."""
    generation_dir = _stale_generation_fixture(
        tmp_path,
        manifest_training_status="done",
        status="done",
        evaluator_results={"mlp_41": {"final_loss": 1.0}},
    )

    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    assert not state.is_stale
    assert state.reason == "training_done"


def test_training_status_with_non_empty_results_is_not_deleted(
    tmp_path: Path,
) -> None:
    """Partial evaluator results should block automatic status removal."""
    generation_dir = _stale_generation_fixture(
        tmp_path,
        evaluator_results={"mlp_41": {"final_loss": 1.0}},
    )

    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    assert not state.is_stale
    assert state.reason == "evaluator_results_present"


def test_recovery_backs_up_removes_stale_files_and_resets_manifest(
    tmp_path: Path,
) -> None:
    """Recovery should backup first, then make the generation claimable again."""
    generation_dir = _stale_generation_fixture(tmp_path)
    dataset_status_path = generation_dir / "dataset_status.json"
    dataset_status_path.write_text('{"status": "done"}\n', encoding="utf-8")
    model_artifact = generation_dir / "models" / "param.pt"
    model_artifact.parent.mkdir(parents=True)
    model_artifact.write_bytes(b"model")
    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    recovered = recover_stale_training_state(generation_dir, state)

    assert recovered
    backups = list(generation_dir.glob("stale_training_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "training_claim.json").is_file()
    assert (backups[0] / "training_status.json").is_file()
    assert not (generation_dir / "training_claim.json").exists()
    assert not (generation_dir / "training_status.json").exists()
    manifest = load_pipeline_manifest(generation_dir / "manifest.json")
    assert manifest.training_status == "not_started"
    assert manifest.metadata["auto_recovery"]["old_training_status"] == "training"
    assert dataset_status_path.is_file()
    assert model_artifact.is_file()


def test_recovery_dry_run_does_not_mutate(tmp_path: Path) -> None:
    """Dry-run recovery should report planned action without touching files."""
    generation_dir = _stale_generation_fixture(tmp_path)
    state = inspect_training_state_for_recovery(
        generation_dir,
        now_utc=_NOW,
        stale_grace_seconds=300,
    )

    recovered = recover_stale_training_state(generation_dir, state, dry_run=True)

    assert recovered
    assert (generation_dir / "training_claim.json").is_file()
    assert (generation_dir / "training_status.json").is_file()
    assert not list(generation_dir.glob("stale_training_backup_*"))
    assert load_pipeline_manifest(generation_dir / "manifest.json").training_status == (
        "training"
    )


def test_training_recovery_cli_dry_run_prints_stale_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI dry-run should inspect without mutation."""
    _stale_generation_fixture(tmp_path)

    exit_code = main([
        "--work-dir",
        str(tmp_path),
        "--generation",
        "38",
        "--dry-run",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "stale=true" in output
    assert "reason=expired_training_claim_empty_evaluator_results" in output
    assert "planned_action=reset_training_to_not_started" in output


def test_training_recovery_cli_recover_patches_files(tmp_path: Path) -> None:
    """CLI recover should safely reset stale training state."""
    generation_dir = _stale_generation_fixture(tmp_path)

    exit_code = main([
        "--work-dir",
        str(tmp_path),
        "--generation",
        "38",
        "--recover",
    ])

    assert exit_code == 0
    assert not (generation_dir / "training_claim.json").exists()
    assert load_pipeline_manifest(generation_dir / "manifest.json").training_status == (
        "not_started"
    )


def _stale_generation_fixture(
    work_dir: Path,
    *,
    generation: int = 38,
    expires_at_utc: str = "2026-07-08T14:03:32Z",
    manifest_training_status: str = "training",
    status: str = "training",
    evaluator_results: dict[str, object] | None = None,
) -> Path:
    generation_dir = work_dir / "pipeline" / f"generation_{generation:06d}"
    generation_dir.mkdir(parents=True, exist_ok=True)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc="2026-07-08T13:00:00Z",
            rows_path=f"rows/generation_{generation:06d}.jsonl",
            dataset_status="done",
            training_status=manifest_training_status,
        ),
        generation_dir / "manifest.json",
    )
    (generation_dir / "training_claim.json").write_text(
        json.dumps({
            "claim_id": "claim-38",
            "claimed_at_utc": "2026-07-08T13:03:32Z",
            "expires_at_utc": expires_at_utc,
            "generation": generation,
            "metadata": {},
            "owner": None,
            "stage": "training",
        })
        + "\n",
        encoding="utf-8",
    )
    (generation_dir / "training_status.json").write_text(
        json.dumps({
            "evaluator_results": {} if evaluator_results is None else evaluator_results,
            "generation": generation,
            "metadata": {},
            "selected_evaluator_name": None,
            "selection_policy": None,
            "status": status,
            "updated_at_utc": "2026-07-08T13:03:32Z",
        })
        + "\n",
        encoding="utf-8",
    )
    return generation_dir
