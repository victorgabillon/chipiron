"""Additional installed-wheel admission checks; no service workers or saved data."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import dataset_publication as publication
from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import save_bootstrap_run_state
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
    setup_fenced,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_takeover_claim_survives_old_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new claimant need not have finished before the old computation is fenced."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    accept = stages.accept_dataset
    before = paths.run_state_path.read_bytes()
    claim_path = paths.pipeline_dataset_claim_path_for_generation(1)
    replacement = []

    def takeover(p: Any, ref: Any) -> Any:
        claim = replace(
            load_pipeline_stage_claim(claim_path), claim_id="replacement-claim"
        )
        save_pipeline_stage_claim(claim, claim_path)
        replacement.append(claim_path.read_bytes())
        return accept(p, ref)

    monkeypatch.setattr(stages, "accept_dataset", takeover)
    with pytest.raises(ValueError, match="Stale Dataset claim"):
        run_dataset(args)
    assert claim_path.read_bytes() == replacement[0]
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.parametrize(
    "marker", [None, {"schema_version": True}, {"schema_version": 99}]
)
def test_protocol_downgrade_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: Any
) -> None:
    """Accepted references cannot silently fall back to unfenced completion semantics."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    run_dataset(args)
    state = load_bootstrap_run_state(paths.run_state_path)
    metadata = dict(state.metadata)
    if marker is None:
        del metadata[publication.KEY]
    else:
        metadata[publication.KEY] = marker
    save_bootstrap_run_state(
        replace(state, metadata=metadata), paths.run_state_path, durable=True
    )
    with pytest.raises(ValueError, match="protocol"):
        run_dataset(args)


@pytest.mark.parametrize("serializer", ["json", "jsonl"])
def test_cooperative_serializers_cannot_overwrite_accepted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serializer: str
) -> None:
    """A caller retaining an accepted path cannot reuse a row serializer to replace it."""
    args, paths, _ = setup_fenced(tmp_path)
    local_leaderboard(monkeypatch, tmp_path)
    result = run_dataset(args)
    path = paths.work_dir / result.rows_path
    before = path.read_bytes()
    with pytest.raises(ValueError, match="immutable Dataset"):
        if serializer == "jsonl":
            save_morpion_supervised_rows_streaming(rows=(), metadata={}, path=path)
        else:
            save_morpion_supervised_rows(
                MorpionSupervisedRows(rows=(), metadata={}), path
            )
    assert path.read_bytes() == before


def test_dataset_acceptance_preserves_actual_reevaluation_proposal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accepted Dataset reference changes neither source identity nor producer cursor."""
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    args, paths, _ = setup_fenced(tmp_path)
    with pytest.MonkeyPatch.context() as adapter:
        atomic_worker(tmp_path, adapter, identifier="same-patch")
    before = paths.pipeline_reevaluation_patch_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.unlink()  # Only this test's unaccepted proposal.
    local_leaderboard(monkeypatch, tmp_path)
    run_dataset(args)
    with pytest.MonkeyPatch.context() as adapter:
        atomic_worker(tmp_path, adapter, identifier="same-patch")
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == before
    assert (
        load_bootstrap_run_state(paths.run_state_path).metadata["atomic_reevaluation"][
            "accepted_cursor"
        ]
        is None
    )
