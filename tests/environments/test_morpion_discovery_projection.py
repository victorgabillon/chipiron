"""Lean discovery is a versioned projection, never checkpoint/receipt authority."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, get_args
from unittest.mock import patch

import pytest

from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    GenerationPublicationError,
    validate_manifest_publication,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    InvalidMorpionPipelineArtifactError,
    MorpionPipelineDatasetStatusArtifact,
    MorpionPipelineStageClaim,
    MorpionPipelineTrainingStatusArtifact,
    load_pipeline_manifest,
    pipeline_dataset_status_from_dict,
    pipeline_dataset_status_to_dict,
    pipeline_manifest_from_dict,
    pipeline_manifest_to_dict,
    pipeline_training_status_from_dict,
    pipeline_training_status_to_dict,
    save_pipeline_manifest,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    dataset_stage_is_pending,
    load_available_pipeline_manifests,
    select_next_claimable_dataset_generation,
    select_next_claimable_training_generation,
    select_next_dataset_generation,
    select_next_training_generation,
    training_stage_is_pending,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    resolve_latest_reevaluation_tree_snapshot,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    publish_scientific_patch,
    run_scientific_publication,
    setup_scientific_baseline,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import publish_fixture_patch

if TYPE_CHECKING:
    from pathlib import Path

KEY = "pipeline_discovery"
RECEIPTS = "checkpoint_bound_reevaluation_v1"


def workspace(root: Path, version: int = 2) -> tuple[Any, Any, Any]:
    """Publish a second tiny generation using the existing scientific fixtures."""
    if version == 2:
        args, paths, _ = setup_scientific_baseline(root)
        publish_scientific_patch(paths)
        state = run_scientific_publication(args, ScientificPublicationRunner())
    else:
        args, paths = setup_publication_baseline(root)
        publish_fixture_patch(paths)
        state = run_publication(args, PublicationRunner())
    return args, paths, state


@pytest.mark.parametrize("version", [1, 2])
def test_new_managed_projection_has_no_historical_ledgers(
    tmp_path: Path, version: int
) -> None:
    """Keep exactly one committed reference and all nonglobal diagnostic fields."""
    _, paths, state = workspace(tmp_path, version)
    original = paths.run_state_path.read_bytes()
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))
    assert PUBLICATION_KEY not in manifest.metadata
    assert RECEIPTS not in manifest.metadata
    assert manifest.metadata[KEY] == {
        "schema_version": 2,
        "generation": 2,
        "publication": state.metadata[PUBLICATION_KEY]["generations"][-1],
        "checkpoint": None,
    }
    assert {k: v for k, v in manifest.metadata.items() if k != KEY} == {
        k: v for k, v in state.metadata.items() if k not in (PUBLICATION_KEY, RECEIPTS)
    }
    validate_manifest_publication(tmp_path, manifest)
    assert paths.run_state_path.read_bytes() == original


@pytest.mark.parametrize("legacy", [False, True])
def test_manifest_and_status_roundtrip(tmp_path: Path, legacy: bool) -> None:
    """Preserve both old opaque ledger metadata and explicitly versioned projections."""
    _, paths, state = workspace(tmp_path)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))
    if legacy:
        manifest = replace(manifest, metadata=state.metadata)
    payload = pipeline_manifest_to_dict(manifest)
    assert pipeline_manifest_to_dict(pipeline_manifest_from_dict(payload)) == payload
    for cls, encode, decode in [
        (
            MorpionPipelineDatasetStatusArtifact,
            pipeline_dataset_status_to_dict,
            pipeline_dataset_status_from_dict,
        ),
        (
            MorpionPipelineTrainingStatusArtifact,
            pipeline_training_status_to_dict,
            pipeline_training_status_from_dict,
        ),
    ]:
        artifact = cls(
            generation=2,
            status="done",
            updated_at_utc="2026-10-21T00:00:00Z",
            metadata=manifest.metadata,
        )
        assert encode(decode(encode(artifact))) == encode(artifact)
    if legacy:
        save_pipeline_manifest(manifest, paths.pipeline_manifest_path_for_generation(2))
        before = paths.pipeline_manifest_path_for_generation(2).read_bytes()
        GrowthPatchReceiptSession(paths, object()).publish_manifest(state)
        assert paths.pipeline_manifest_path_for_generation(2).read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "float_version",
        "mixed",
        "wrong_generation",
        "no_reference",
        "bad_checkpoint",
    ],
)
def test_projection_version_is_not_guessed(tmp_path: Path, change: str) -> None:
    """Reject ambiguous new envelopes on manifest and both status readers."""
    _, paths, _ = workspace(tmp_path)
    metadata = copy.deepcopy(
        load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2)).metadata
    )
    if change == "unknown":
        metadata[KEY]["schema_version"] = 3
    elif change == "float_version":
        metadata[KEY]["schema_version"] = 2.0
    elif change == "mixed":
        metadata[RECEIPTS] = {"receipts": []}
    elif change == "wrong_generation":
        metadata[KEY]["generation"] = 1
    elif change == "no_reference":
        metadata[KEY]["publication"] = None
    else:
        metadata[KEY].update(
            publication=None, checkpoint={"path": "x", "sha256": "bad"}
        )
    for cls in [
        type(load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))),
        MorpionPipelineDatasetStatusArtifact,
        MorpionPipelineTrainingStatusArtifact,
    ]:
        kwargs = {"generation": 2, "metadata": metadata}
        if cls.__name__ == "MorpionPipelineGenerationManifest":
            kwargs["created_at_utc"] = "now"
        else:
            kwargs.update(status="done", updated_at_utc="now")
        with pytest.raises(InvalidMorpionPipelineArtifactError):
            cls(**kwargs)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "failure", ["source_changed", "checkpoint_missing", "bad_path"]
)
def test_existing_capsule_checks_still_run(
    tmp_path: Path, legacy: bool, failure: str
) -> None:
    """Identical dependency failures remain fatal with either discovery representation."""
    _, paths, state = workspace(tmp_path)
    path = paths.pipeline_manifest_path_for_generation(2)
    manifest = load_pipeline_manifest(path)
    if legacy:
        manifest = replace(manifest, metadata=state.metadata)
    if failure == "bad_path":
        manifest = replace(manifest, tree_snapshot_path="wrong.json")
    elif failure == "source_changed":
        source = tmp_path / state.latest_tree_snapshot_path
        source.write_bytes(source.read_bytes() + b" ")
    else:
        (tmp_path / state.latest_runtime_checkpoint_path).unlink()
    save_pipeline_manifest(manifest, path)
    with pytest.raises(ValueError):
        validate_manifest_publication(
            tmp_path, load_available_pipeline_manifests(paths)[2]
        )


def test_projection_cannot_replace_authority(tmp_path: Path) -> None:
    """A plausible forged reference cannot select an uncommitted capsule."""
    _, paths, state = workspace(tmp_path)
    path = paths.pipeline_manifest_path_for_generation(2)
    manifest = load_pipeline_manifest(path)
    metadata = copy.deepcopy(manifest.metadata)
    metadata[KEY]["publication"]["descriptor_sha256"] = "0" * 64
    save_pipeline_manifest(replace(manifest, metadata=metadata), path)
    with pytest.raises(GenerationPublicationError, match="projection"):
        validate_manifest_publication(
            tmp_path, load_available_pipeline_manifests(paths)[2]
        )
    with pytest.raises(ValueError, match="projection"):
        GrowthPatchReceiptSession(paths, object()).publish_manifest(state)


@pytest.mark.parametrize("claimed", [False, True])
def test_selected_order_and_discovery_match_legacy(
    tmp_path: Path, claimed: bool
) -> None:
    """The same claim/status matrix yields the same Dataset/Training/Reevaluation choices."""
    _, paths, state = workspace(tmp_path)
    current = load_available_pipeline_manifests(paths)
    if claimed:
        for stage, path in (
            ("dataset", paths.pipeline_dataset_claim_path_for_generation(2)),
            ("training", paths.pipeline_training_claim_path_for_generation(2)),
        ):
            save_pipeline_stage_claim(
                MorpionPipelineStageClaim(
                    generation=2,
                    stage=stage,
                    claim_id="synthetic-active-claim",
                    claimed_at_utc="2000-01-01T00:00:00Z",
                    expires_at_utc="2100-01-01T00:00:00Z",
                ),
                path,
            )
    legacy = {
        g: replace(m, metadata={k: v for k, v in m.metadata.items() if k != KEY})
        for g, m in current.items()
    }
    # Actual historical metadata is still accepted without the new marker.
    legacy[2] = replace(legacy[2], metadata=state.metadata)
    for dataset in ["not_started", "extracting_rows", "done", "failed"]:
        for training in ["not_started", "training", "done", "failed"]:
            projections = []
            for manifests in (current, legacy):
                candidates = {
                    g: replace(
                        m,
                        rows_path="rows.jsonl",
                        dataset_status=dataset,
                        training_status=training,
                    )
                    for g, m in manifests.items()
                }
                projections.append((
                    [dataset_stage_is_pending(m) for m in candidates.values()],
                    [training_stage_is_pending(m) for m in candidates.values()],
                    select_next_dataset_generation(candidates),
                    select_next_training_generation(candidates),
                    select_next_training_generation(
                        candidates, lower_bound_generation=1
                    ),
                    select_next_claimable_dataset_generation(
                        paths, candidates, now_unix_s=1e9
                    ),
                    select_next_claimable_training_generation(
                        paths, candidates, now_unix_s=1e9
                    ),
                ))
            assert projections[0] == projections[1]
    expected = resolve_latest_reevaluation_tree_snapshot(paths)
    for g, manifest in legacy.items():
        save_pipeline_manifest(manifest, paths.pipeline_manifest_path_for_generation(g))
    assert resolve_latest_reevaluation_tree_snapshot(paths) == expected
    assert expected == (2, tmp_path / state.latest_tree_snapshot_path)


def test_manifest_repair_and_durable_stage_rewrite(tmp_path: Path) -> None:
    """Reconstruct from the committed state, never from a stale temporary projection."""
    _, paths, state = workspace(tmp_path)
    path = paths.pipeline_manifest_path_for_generation(2)
    original = load_pipeline_manifest(path)
    path.unlink()
    path.with_suffix(".json.tmp").write_text('{"generation":999')
    before = paths.run_state_path.read_bytes()
    GrowthPatchReceiptSession(paths, object()).publish_manifest(state)
    assert load_pipeline_manifest(path) == original
    with patch(
        "chipiron.environments.morpion.bootstrap.derived.provenance.atomic_json",
        wraps=atomic_json,
    ) as durable:
        save_pipeline_manifest(replace(original, dataset_status="done"), path)
    assert durable.call_count == 1
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_actual_dataset_worker_rows_match_old_discovery(
    tmp_path: Path, policy: str
) -> None:
    """Run both projection representations on the same immutable scientific source."""
    from chipiron.environments.morpion.bootstrap.pipeline.stages import (
        run_pipeline_dataset_stage,
    )

    args, paths, state = workspace(tmp_path)
    args = replace(
        args,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
    )
    path = paths.pipeline_manifest_path_for_generation(2)
    new = load_pipeline_manifest(path)
    observations = []
    for metadata in (state.metadata, new.metadata):
        save_pipeline_manifest(replace(new, metadata=metadata), path)
        if policy.startswith("pv_exact_then_"):
            with pytest.raises(ValueError, match="gates"):
                run_pipeline_dataset_stage(
                    args,
                    generation=2,
                    use_streaming_sharded_reader=True,
                    allow_scientific_v2=True,
                    reader_cache_dir=tmp_path / "cache",
                )
            continue
        result = run_pipeline_dataset_stage(
            args,
            generation=2,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
            reader_cache_dir=tmp_path / "cache",
        )
        assert result.rows_path is not None
        row_bytes = (tmp_path / result.rows_path).read_bytes()
        data = pipeline_manifest_to_dict(result)
        # Only the explicitly versioned projection and wall-clock completion time differ.
        data["metadata"] = {
            k: v
            for k, v in result.metadata.items()
            if k not in {KEY, PUBLICATION_KEY, RECEIPTS, "dataset_completed_at_utc"}
        }
        observations.append((row_bytes, data))
    if observations:
        assert observations[0] == observations[1]


def test_status_writers_and_dashboard_match_legacy(tmp_path: Path) -> None:
    """Real status writers retain dashboard inputs without copying authoritative history."""
    from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
        load_performance_summary,
    )
    from chipiron.environments.morpion.bootstrap.pipeline.manifests import (
        latest_prior_dataset_status_artifact,
        save_dataset_manifest_status,
        save_training_manifest_status,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_training_status_file,
    )

    _, paths, state = workspace(tmp_path)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2))
    summaries = []
    for metadata in (state.metadata, manifest.metadata):
        selected = replace(manifest, metadata={**metadata, "dataset_rows": 2})
        selected = save_dataset_manifest_status(
            paths=paths, manifest=selected, dataset_status="done", timestamp_utc="now"
        )
        selected = save_training_manifest_status(
            paths=paths,
            manifest=selected,
            training_status="failed",
            timestamp_utc="now",
        )
        dataset = latest_prior_dataset_status_artifact(paths=paths, generation=3)
        training = load_pipeline_training_status_file(
            paths.pipeline_training_status_path_for_generation(2)
        )
        assert dataset is not None and dataset.generation == 2
        assert dataset.status == "done" and training.status == "failed"
        assert dataset.metadata == training.metadata == selected.metadata
        summaries.append(load_performance_summary(tmp_path))
    assert summaries[0] == summaries[1]


@pytest.mark.parametrize("legacy", [False, True])
def test_stale_training_recovery_preserves_projection(
    tmp_path: Path, legacy: bool
) -> None:
    """Recovery changes stage state and diagnostics without recreating removed ledgers."""
    from datetime import UTC, datetime

    from chipiron.environments.morpion.bootstrap.pipeline.manifests import (
        save_training_manifest_status,
    )
    from chipiron.environments.morpion.bootstrap.pipeline.training_recovery import (
        inspect_training_state_for_recovery,
        recover_stale_training_state,
    )

    _, paths, state = workspace(tmp_path)
    path = paths.pipeline_manifest_path_for_generation(2)
    manifest = load_pipeline_manifest(path)
    manifest = replace(manifest, rows_path="rows.jsonl", dataset_status="done")
    if legacy:
        manifest = replace(manifest, metadata=state.metadata)
    save_training_manifest_status(
        paths=paths,
        manifest=manifest,
        training_status="training",
        timestamp_utc="2000-01-01T00:00:00Z",
    )
    save_pipeline_stage_claim(
        MorpionPipelineStageClaim(
            generation=2,
            stage="training",
            claim_id="expired",
            claimed_at_utc="2000-01-01T00:00:00Z",
            expires_at_utc="2000-01-01T01:00:00Z",
        ),
        paths.pipeline_training_claim_path_for_generation(2),
    )
    original_head = paths.run_state_path.read_bytes()
    stale = inspect_training_state_for_recovery(
        path.parent, now_utc=datetime(2026, 10, 21, tzinfo=UTC), stale_grace_seconds=300
    )
    assert stale.reason == "expired_training_claim_empty_evaluator_results"
    assert recover_stale_training_state(path.parent, stale)
    recovered = load_pipeline_manifest(path)
    assert recovered.training_status == "not_started"
    assert {
        k: v for k, v in recovered.metadata.items() if k != "auto_recovery"
    } == manifest.metadata
    assert paths.run_state_path.read_bytes() == original_head
    assert not paths.pipeline_training_status_path_for_generation(2).exists()
    assert select_next_claimable_training_generation(paths, {2: recovered}) == 2
