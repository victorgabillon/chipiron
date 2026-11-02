"""Discovery can overlap the durable commit and its later advisory projection."""

from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    publication_records,
    validate_manifest_publication,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    InvalidMorpionPipelineArtifactError,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
)
from tests.environments.morpion_atomic_reevaluation_helpers import publish_proposal
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
    setup_fenced,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.run_state import (
        MorpionBootstrapRunState,
    )


@pytest.mark.parametrize("accepted", [False, True])
def test_discovery_during_commit_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    """Pause the real publisher at the exact failed-run boundary, without sleeps."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    if accepted:
        run_dataset(args)
    publish_proposal(paths)
    publish = GrowthPatchReceiptSession.publish_manifest
    observed = []

    def at_boundary(
        session: GrowthPatchReceiptSession, state: MorpionBootstrapRunState
    ) -> None:
        if state.generation == 2:
            assert load_bootstrap_run_state(paths.run_state_path).generation == 2
            assert not paths.pipeline_manifest_path_for_generation(2).exists()
            before = paths.run_state_path.read_bytes()
            discovered = load_available_pipeline_manifests(paths)
            assert set(discovered) == {1}
            assert discovered[1].dataset_status == (
                "done" if accepted else "not_started"
            )
            validate_manifest_publication(tmp_path, discovered[1])
            assert paths.run_state_path.read_bytes() == before
            assert not paths.pipeline_manifest_path_for_generation(2).exists()
            observed.append(2)
        publish(session, state)

    monkeypatch.setattr(GrowthPatchReceiptSession, "publish_manifest", at_boundary)
    run_closed_publication(args, ClosedPublicationRunner())
    assert observed == [2]
    discovered = load_available_pipeline_manifests(paths)
    assert set(discovered) == {1, 2}
    validate_manifest_publication(tmp_path, discovered[2])


@pytest.mark.parametrize("accepted", [False, True])
def test_absent_projection_does_not_create_or_lose_accepted_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    """Unavailable pending work waits; accepted immutable evidence still projects."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    expected = run_dataset(args) if accepted else None
    path = paths.pipeline_manifest_path_for_generation(1)
    path.unlink()
    before = paths.run_state_path.read_bytes()
    discovered = load_available_pipeline_manifests(paths)
    assert paths.run_state_path.read_bytes() == before and not path.exists()
    if accepted:
        assert expected is not None
        assert discovered[1].rows_path == expected.rows_path
        assert discovered[1].dataset_status == "done"
    else:
        assert not discovered
        GrowthPatchReceiptSession(paths, object()).publish_manifest(
            load_bootstrap_run_state(paths.run_state_path)
        )
        assert (
            load_available_pipeline_manifests(paths)[1].dataset_status == "not_started"
        )


@pytest.mark.parametrize("damage", ["malformed_manifest", "accepted_proposal"])
def test_discovery_corruption_is_still_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Deferring an absent advisory file never suppresses corrupt visible evidence."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    if damage == "accepted_proposal":
        run_dataset(args)
        state = load_bootstrap_run_state(paths.run_state_path)
        record = publication_records(state.metadata)[0]
        attempt = record["accepted_dataset"]["attempt_id"]
        path = tmp_path / "dataset_publications_v1/objects" / attempt / "proposal.json"
        path.write_text("{}")
        paths.pipeline_manifest_path_for_generation(1).unlink()
    else:
        paths.pipeline_manifest_path_for_generation(1).write_text("{}")
    with pytest.raises((
        GenerationPublicationError,
        InvalidMorpionPipelineArtifactError,
    )):
        load_available_pipeline_manifests(paths)
