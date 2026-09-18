"""Tests for the Morpion pipeline stage-claim protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    PipelineStageAlreadyClaimedError,
    PipelineStageClaimMismatchError,
    claim_pipeline_stage,
    load_active_pipeline_stage_claim,
    load_pipeline_stage_claim,
    release_pipeline_stage_claim,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_claim_roundtrip(tmp_path: Path) -> None:
    """Claims should save, load, and release cleanly."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    claim_path = paths.pipeline_dataset_claim_path_for_generation(1)

    claim = claim_pipeline_stage(
        generation=1,
        stage="dataset",
        claim_path=claim_path,
        now_unix_s=1000.0,
        claim_id="claim-1",
        owner="worker-a",
        metadata={"entrypoint": "test"},
    )

    assert claim_path.is_file()
    assert load_pipeline_stage_claim(claim_path) == claim
    assert claim.generation == 1
    assert claim.stage == "dataset"
    assert claim.owner == "worker-a"
    assert claim.metadata == {"entrypoint": "test"}

    release_pipeline_stage_claim(claim_path=claim_path, claim_id=claim.claim_id)

    assert not claim_path.exists()


def test_non_expired_claim_blocks_second_claimant(tmp_path: Path) -> None:
    """Active claims should reject concurrent claim attempts."""
    claim_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).pipeline_dataset_claim_path_for_generation(1)
    claim_pipeline_stage(
        generation=1,
        stage="dataset",
        claim_path=claim_path,
        now_unix_s=1000.0,
        claim_id="first",
    )

    with pytest.raises(
        PipelineStageAlreadyClaimedError,
        match=r"stage=dataset generation=1 claim_id=first",
    ):
        claim_pipeline_stage(
            generation=1,
            stage="dataset",
            claim_path=claim_path,
            now_unix_s=1005.0,
            claim_id="second",
        )


def test_expired_claim_can_be_replaced(tmp_path: Path) -> None:
    """Expired claims should be replaceable by a new worker."""
    claim_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).pipeline_training_claim_path_for_generation(1)
    first = claim_pipeline_stage(
        generation=1,
        stage="training",
        claim_path=claim_path,
        now_unix_s=1000.0,
        ttl_seconds=10.0,
        claim_id="first",
    )

    second = claim_pipeline_stage(
        generation=1,
        stage="training",
        claim_path=claim_path,
        now_unix_s=1011.0,
        ttl_seconds=10.0,
        claim_id="second",
    )

    assert first.claim_id == "first"
    assert second.claim_id == "second"
    assert load_pipeline_stage_claim(claim_path).claim_id == "second"


def test_release_mismatch_does_not_delete(tmp_path: Path) -> None:
    """Mismatched releases should fail without deleting the active claim."""
    claim_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).pipeline_dataset_claim_path_for_generation(1)
    claim = claim_pipeline_stage(
        generation=1,
        stage="dataset",
        claim_path=claim_path,
        claim_id="first",
    )

    with pytest.raises(PipelineStageClaimMismatchError):
        release_pipeline_stage_claim(claim_path=claim_path, claim_id="second")

    assert claim_path.exists()
    assert load_pipeline_stage_claim(claim_path).claim_id == claim.claim_id


def test_load_active_pipeline_stage_claim(tmp_path: Path) -> None:
    """Active-claim loading should hide missing and expired claims."""
    claim_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).pipeline_training_claim_path_for_generation(1)

    assert load_active_pipeline_stage_claim(claim_path) is None

    claim = claim_pipeline_stage(
        generation=1,
        stage="training",
        claim_path=claim_path,
        now_unix_s=1000.0,
        ttl_seconds=10.0,
        claim_id="first",
    )

    assert load_active_pipeline_stage_claim(claim_path, now_unix_s=1005.0) == claim
    assert load_active_pipeline_stage_claim(claim_path, now_unix_s=1011.0) is None


def test_invalid_claim_ttl(tmp_path: Path) -> None:
    """Invalid claim TTLs should fail clearly."""
    claim_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).pipeline_dataset_claim_path_for_generation(1)

    with pytest.raises(ValueError, match="claim ttl_seconds must be > 0"):
        claim_pipeline_stage(
            generation=1,
            stage="dataset",
            claim_path=claim_path,
            ttl_seconds=0.0,
        )
