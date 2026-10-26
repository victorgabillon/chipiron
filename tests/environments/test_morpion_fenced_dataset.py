"""Acceptance belongs to run_state, never a visible output or TTL claim."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import dataset_publication as publication
from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline.manifests import (
    pinned_generation_manifest,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    save_pipeline_manifest,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
    setup_fenced,
)


def reference(paths: Any, generation: int = 1) -> Any:
    """Read the result only from the existing committed generation entry."""
    return next(
        r
        for r in publication_records(
            load_bootstrap_run_state(paths.run_state_path).metadata
        )
        if r["generation"] == generation
    )[publication.ACCEPTED]


@pytest.mark.fast
def test_normal_restart_and_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact retry cannot rewrite rows, receipt history or the accepted cursor."""
    args, paths, initial = setup_fenced(tmp_path)
    args = replace(args, use_backed_up_value=False)
    local_leaderboard(monkeypatch, tmp_path)
    result = run_dataset(args)
    identity = reference(paths)
    before = paths.run_state_path.read_bytes()
    rows = (tmp_path / result.rows_path).read_bytes()
    assert run_dataset(args) == result
    assert publication.accept_dataset(paths, identity) == result
    assert paths.run_state_path.read_bytes() == before
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.generation == initial.generation
    assert {k: v for k, v in state.metadata.items() if k != PUBLICATION_KEY} == {
        k: v for k, v in initial.metadata.items() if k != PUBLICATION_KEY
    }
    assert (tmp_path / result.rows_path).read_bytes() == rows
    with pytest.raises(ValueError, match="Conflicting Dataset configuration"):
        run_dataset(replace(args, max_rows=0))
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.fast
def test_unaccepted_visible_rows_cannot_feed_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forged completion projections are ignored when no accepted reference exists."""
    args, paths, _ = setup_fenced(tmp_path)
    from tests.environments.morpion_atomic_reevaluation_helpers import publish_proposal
    from tests.environments.morpion_durable_closed_helpers import (
        ClosedPublicationRunner,
        run_closed_publication,
    )

    publish_proposal(paths)
    run_closed_publication(args, ClosedPublicationRunner())
    manifest_path = paths.pipeline_manifest_path_for_generation(2)
    manifest = load_pipeline_manifest(manifest_path)
    rows = paths.rows_jsonl_path_for_generation(2)
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_text("{}\n")
    save_pipeline_manifest(
        replace(
            manifest, dataset_status="done", rows_path=paths.relative_to_work_dir(rows)
        ),
        manifest_path,
    )
    assert load_available_pipeline_manifests(paths)[2].dataset_status == "not_started"
    with pytest.raises(ValueError, match="dataset_status"):
        stages.run_pipeline_training_stage(args, generation=2)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rows",
        "changed_rows",
        "changed_descriptor",
        "missing_manifest",
        "stale_status",
    ],
)
def test_recovery_projections_and_artifact_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Repair projections from committed evidence; missing/corrupt dependencies stop."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    result = run_dataset(args)
    rows = tmp_path / result.rows_path
    before = paths.run_state_path.read_bytes()
    if damage == "missing_rows":
        rows.unlink()
    elif damage == "changed_rows":
        rows.write_bytes(rows.read_bytes() + b" ")
    elif damage == "changed_descriptor":
        (rows.parent / "proposal.json").write_text("{}")
    elif damage == "missing_manifest":
        paths.pipeline_manifest_path_for_generation(1).unlink()
        paths.pipeline_dataset_status_path_for_generation(1).unlink()
    else:
        save_pipeline_manifest(
            replace(result, dataset_status="failed", rows_path="wrong"),
            paths.pipeline_manifest_path_for_generation(1),
        )
        paths.pipeline_dataset_status_path_for_generation(1).write_text("{}")
    if damage in {"missing_rows", "changed_rows", "changed_descriptor"}:
        with (
            pytest.raises(ValueError),
            pinned_generation_manifest(paths=paths, generation=1),
        ):
            pytest.fail("Corrupted accepted dependency was exposed")
    else:
        assert run_dataset(args).rows_path == result.rows_path
        assert (
            load_pipeline_manifest(
                paths.pipeline_manifest_path_for_generation(1)
            ).dataset_status
            == "done"
        )
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.fast
def test_expired_old_worker_cannot_replace_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TTL takeover finishes while the old worker still holds its source pin."""
    args, paths, _ = setup_fenced(tmp_path)
    args = replace(args, use_backed_up_value=False)
    local_leaderboard(monkeypatch, tmp_path)
    writer = stages.save_morpion_supervised_rows_streaming
    winner = []

    def interleave(**kwargs: Any) -> Any:
        claim_path = paths.pipeline_dataset_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim_path),
                expires_at_utc="2000-01-01T00:00:00Z",
            ),
            claim_path,
        )
        monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", writer)
        completed = run_dataset(args, claim_owner="new")
        winner.append((
            reference(paths),
            completed.rows_path,
            (tmp_path / completed.rows_path).read_bytes(),
        ))
        return writer(**kwargs)

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", interleave)
    with pytest.raises(ValueError, match="Conflicting accepted"):
        run_dataset(replace(args, max_rows=0), claim_owner="old")
    assert reference(paths) == winner[0][0]
    assert (tmp_path / winner[0][1]).read_bytes() == winner[0][2]
    assert load_available_pipeline_manifests(paths)[1].rows_path == winner[0][1]
    assert len(list((tmp_path / publication.NAMESPACE / "objects").iterdir())) == 2


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_epoch",
        "foreign_workspace",
        "source_identity",
        "model_identity",
        "wrong_generation",
        "bad_completion",
        "changed_output",
        "missing_output",
    ],
)
def test_invalid_proposals_never_enter_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Corruption and foreign computation fail before the only authoritative replace."""
    from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json

    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    accept = stages.accept_dataset
    before = paths.run_state_path.read_bytes()

    def corrupt(p: Any, ref: Any) -> Any:
        directory = tmp_path / publication.NAMESPACE / "objects" / ref["attempt_id"]
        proposal_path = directory / "proposal.json"
        data = read_object(proposal_path)
        admission = data["admission"]
        if fault == "foreign_epoch":
            admission["identity"]["epoch_id"] = "f" * 32
        elif fault == "foreign_workspace":
            admission["identity"]["workspace_id"] = "f" * 32
        elif fault == "source_identity":
            admission["source"]["descriptor_sha256"] = "0" * 64
        elif fault == "model_identity":
            admission["source_model_pins"] = []
        elif fault == "wrong_generation":
            admission["generation"] = 999
        elif fault == "bad_completion":
            data["manifest"]["generation"] = 999
        elif fault == "changed_output":
            (directory / "rows.jsonl").write_bytes(b"broken\n")
        else:
            (directory / "rows.jsonl").unlink()
        # Rehash a foreign-but-well-formed proposal to test binding, not just its digest.
        atomic_json(proposal_path, data)
        return accept(
            p, {**ref, "descriptor_sha256": publication.file_digest(proposal_path)}
        )

    monkeypatch.setattr(stages, "accept_dataset", corrupt)
    with pytest.raises(ValueError):
        run_dataset(args)
    assert paths.run_state_path.read_bytes() == before
    assert load_available_pipeline_manifests(paths)[1].dataset_status == "not_started"


@pytest.mark.parametrize("content", [b"{bad-json}\n", b"", b"{}\n"])
def test_malformed_private_rows_are_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """Use the existing row parser to refuse syntactically incomplete private output."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    writer = stages.save_morpion_supervised_rows_streaming
    before = paths.run_state_path.read_bytes()

    def damaged(**kwargs: Any) -> Any:
        stats = writer(**kwargs)
        stats.path.write_bytes(content)
        return stats

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", damaged)
    with pytest.raises((ValueError, TypeError)):
        run_dataset(args)
    assert paths.run_state_path.read_bytes() == before
    assert not (tmp_path / publication.NAMESPACE / "objects").exists()


def test_growth_can_advance_while_old_source_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Growth is not locked during extraction; G1 remains eligible after G2 commits."""
    from tests.environments.morpion_atomic_reevaluation_helpers import publish_proposal
    from tests.environments.morpion_durable_closed_helpers import (
        ClosedPublicationRunner,
        run_closed_publication,
    )

    args, paths, _initial = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    writer = stages.save_morpion_supervised_rows_streaming

    def advance(**kwargs: Any) -> Any:
        publish_proposal(paths)
        run_closed_publication(args, ClosedPublicationRunner())
        assert load_bootstrap_run_state(paths.run_state_path).generation == 2
        monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", writer)
        return writer(**kwargs)

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", advance)
    older = run_dataset(args)
    assert older.generation == 1
    assert reference(paths)
    after = load_bootstrap_run_state(paths.run_state_path)
    assert after.generation == 2
    assert after.metadata["atomic_reevaluation"]["accepted_cursor"] is not None
    assert after.metadata["checkpoint_bound_reevaluation_v1"]["receipts"]
    run_dataset(args, generation=2)
    assert reference(paths, 2)
    # Later Growth must preserve both accepted entries and never copy them into discovery.
    publish_proposal(paths, identifier="proposal-2")
    run_closed_publication(args, ClosedPublicationRunner())
    assert reference(paths) and reference(paths, 2)
    assert load_bootstrap_run_state(paths.run_state_path).generation == 3
    assert "accepted_dataset" not in str(
        load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(3)).metadata
    )


def test_source_pin_and_growth_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real kernel pins span computation; the exclusive owner only gates admission/commit."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        capsule_lease,
        growth_receipt_ownership,
    )

    args, paths, initial = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    identifier = publication_records(initial.metadata)[0]["id"]
    writer = stages.save_morpion_supervised_rows_streaming

    def check_pin(**kwargs: Any) -> Any:
        with (
            pytest.raises(BlockingIOError),
            capsule_lease(
                tmp_path, identifier, version=3, exclusive=True, nonblocking=True
            ),
        ):
            pytest.fail("Scientific source was not pinned")
        with growth_receipt_ownership(paths):
            pass
        return writer(**kwargs)

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", check_pin)
    run_dataset(args)
    with capsule_lease(
        tmp_path, identifier, version=3, exclusive=True, nonblocking=True
    ):
        pass
    with growth_receipt_ownership(paths), pytest.raises(BlockingIOError):
        run_dataset(args)


@pytest.mark.parametrize("fault", ["science", "model", "descriptor", "epoch_switch"])
def test_source_and_epoch_changed_during_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Never retag old computation with a newer epoch or accept mutated source bytes."""
    from chipiron.environments.morpion.bootstrap.run_state import (
        save_bootstrap_run_state,
    )

    args, paths, initial = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    writer = stages.save_morpion_supervised_rows_streaming
    record = publication_records(initial.metadata)[0]
    descriptor_path = tmp_path / record["descriptor_path"]
    descriptor = read_object(descriptor_path)
    before = paths.run_state_path.read_bytes()
    mutated_authority = []

    def change(**kwargs: Any) -> Any:
        result = writer(**kwargs)
        if fault == "epoch_switch":
            data = {**initial.metadata["atomic_reevaluation"], "epoch_id": "f" * 32}
            save_bootstrap_run_state(
                replace(
                    initial, metadata={**initial.metadata, "atomic_reevaluation": data}
                ),
                paths.run_state_path,
                durable=True,
            )
            mutated_authority.append(paths.run_state_path.read_bytes())
        else:
            if fault == "science":
                target = (
                    tmp_path / record["tree_snapshot_path"]
                ).parent / "updates.jsonl"
            elif fault == "model":
                target = next(
                    (
                        descriptor_path.parent / descriptor["model_pins"][0]["path"]
                    ).rglob("param.pt")
                )
            else:
                target = descriptor_path
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", change)
    with pytest.raises(ValueError):
        run_dataset(args)
    assert paths.run_state_path.read_bytes() == (
        mutated_authority[0] if mutated_authority else before
    )
    assert (
        publication.ACCEPTED
        not in publication_records(
            load_bootstrap_run_state(paths.run_state_path).metadata
        )[0]
    )


def test_live_owner_releases_its_claim_after_contended_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live attempt can release its exact claim; an expired unknown owner cannot."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Timer

    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        growth_receipt_ownership,
    )

    args, paths, _ = setup_fenced(tmp_path)
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "2")
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    held, release = Event(), Event()

    def competing_growth() -> None:
        with growth_receipt_ownership(paths):
            held.set()
            assert release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with publication.admit_dataset(
            paths, args, manifest, ttl_seconds=60, owner=None
        ):
            claim = load_pipeline_stage_claim(
                paths.pipeline_dataset_claim_path_for_generation(1)
            )
            assert claim.owner is None
            result = executor.submit(competing_growth)
            assert held.wait(1)
            timer = Timer(0.1, release.set)
            timer.start()
        result.result(timeout=2)
        timer.join()
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()
