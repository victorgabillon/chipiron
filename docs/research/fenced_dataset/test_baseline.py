"""Characterize the pre-fence Dataset acceptance gaps with new tiny fixtures."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_manifest,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    select_next_dataset_generation,
    training_stage_is_pending,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_atomic_reevaluation_helpers import (
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)


def adapter(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Use the existing explicit closed reader and redirect optional leaderboard I/O."""
    monkeypatch.setattr(stages, "is_sharded_generation_manifest", lambda _: True)
    monkeypatch.setattr(
        stages,
        "ShardedTrainingGenerationView",
        lambda *a, **kw: ShardedTrainingGenerationView(
            *a, **{**kw, "allow_closed_snapshot": True}
        ),
    )
    persist = stages.persist_certified_leaderboard_candidates
    monkeypatch.setattr(
        stages,
        "persist_certified_leaderboard_candidates",
        lambda **kwargs: persist(**kwargs, leaderboard_path=root / "leaderboard.jsonl"),
    )


def test_completed_output_can_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call with different filtering overwrites an apparent completion."""
    args, paths, _ = setup_atomic(tmp_path)
    args = replace(args, use_backed_up_value=False)
    adapter(monkeypatch, tmp_path)
    authority = paths.run_state_path.read_bytes()
    first = stages.run_pipeline_dataset_stage(
        args, generation=1, use_streaming_sharded_reader=True, allow_scientific_v2=True
    )
    rows = paths.work_dir / first.rows_path
    before = rows.read_bytes()
    second = stages.run_pipeline_dataset_stage(
        replace(args, max_rows=0),
        generation=1,
        use_streaming_sharded_reader=True,
        allow_scientific_v2=True,
    )
    assert first.dataset_status == second.dataset_status == "done"
    assert before != rows.read_bytes()
    assert paths.run_state_path.read_bytes() == authority


def test_visible_completion_is_eligible_without_accepted_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completion projection alone drives Training selection, even after output loss."""
    args, paths, _ = setup_atomic(tmp_path)
    args = replace(args, use_backed_up_value=False)
    adapter(monkeypatch, tmp_path)
    manifest = stages.run_pipeline_dataset_stage(
        args, generation=1, use_streaming_sharded_reader=True, allow_scientific_v2=True
    )
    assert training_stage_is_pending(manifest)
    (paths.work_dir / manifest.rows_path).unlink()
    assert training_stage_is_pending(
        load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    )


def test_selection_backfills_older_generation(tmp_path: Path) -> None:
    """Existing latest-pending scheduling deliberately has no Dataset watermark."""
    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths)
    run_closed_publication(args, ClosedPublicationRunner())
    manifests = {
        g: load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(g))
        for g in (1, 2)
    }
    assert select_next_dataset_generation(manifests) == 2
    manifests[2] = replace(manifests[2], dataset_status="done")
    save_pipeline_manifest(manifests[2], paths.pipeline_manifest_path_for_generation(2))
    assert select_next_dataset_generation(manifests) == 1


def test_expired_old_worker_overwrites_takeover_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interleave two real stage invocations at the writer, without service workers."""
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_stage_claim,
        save_pipeline_stage_claim,
    )

    args, paths, _ = setup_atomic(tmp_path)
    args = replace(args, use_backed_up_value=False)
    adapter(monkeypatch, tmp_path)
    writer = stages.save_morpion_supervised_rows_streaming
    takeover_bytes = []
    authority = paths.run_state_path.read_bytes()

    def suspend_old_worker(**kwargs: Any) -> Any:
        claim_path = paths.pipeline_dataset_claim_path_for_generation(1)
        claim = load_pipeline_stage_claim(claim_path)
        save_pipeline_stage_claim(
            replace(claim, expires_at_utc="2000-01-01T00:00:00Z"), claim_path
        )
        # Let the replacement finish while the old worker retains its source pin.
        monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", writer)
        replacement = stages.run_pipeline_dataset_stage(
            args,
            generation=1,
            claim_owner="replacement",
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
        )
        takeover_bytes.append((paths.work_dir / replacement.rows_path).read_bytes())
        return writer(**kwargs)

    monkeypatch.setattr(
        stages, "save_morpion_supervised_rows_streaming", suspend_old_worker
    )
    old = stages.run_pipeline_dataset_stage(
        replace(args, max_rows=0),
        generation=1,
        claim_owner="expired-old-worker",
        use_streaming_sharded_reader=True,
        allow_scientific_v2=True,
    )
    assert old.dataset_status == "done"
    assert (paths.work_dir / old.rows_path).read_bytes() != takeover_bytes[0]
    assert paths.run_state_path.read_bytes() == authority
