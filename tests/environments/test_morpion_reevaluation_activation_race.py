"""A real committed activation supersedes unfinished work, never publishes it."""

import json
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_model_activation_edges import propose

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "boundary", ["admission", "model_load", "inference", "publication"]
)
def test_committed_activation_discards_old_work_and_next_job_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Insert a real activation at each boundary; no sleeps or invented model identity."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    activated = []
    target, name = {
        "admission": (worker, "resolve_latest_reevaluation_tree_snapshot"),
        "model_load": (worker, "build_active_model_reevaluation_evaluator"),
        "inference": (
            worker.MorpionActiveModelNodeReevaluationEvaluator,
            "evaluate_node_window",
        ),
        "publication": (worker, "proposal_patch"),
    }[boundary]
    original = getattr(target, name)

    def activate() -> None:
        assert not activated
        state = activate_model(paths, request)
        activated.append((state, paths.run_state_path.read_bytes()))

    def overlap(*positional: Any, **keywords: Any) -> Any:
        if boundary == "model_load":
            activate()
        result = original(*positional, **keywords)
        if boundary != "model_load":
            activate()
        return result

    with monkeypatch.context() as injection:
        injection.setattr(target, name, overlap)
        result = propose(args)
    assert not result.patch_written and result.reason == "model_activated"
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    state, authority = activated[0]
    assert paths.run_state_path.read_bytes() == authority
    assert not receipt_ledger(state)
    assert list((paths.pipeline_dir / "performance").rglob("*discarded*"))
    # An ordinary next job uses M2, then ordinary Growth consumes its real patch.
    assert propose(args).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert patch.metadata["atomic_reevaluation"]["model"] == request["candidate"]
    runner, _ = _runner(paths, state)
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=1,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    grown = load_bootstrap_run_state(paths.run_state_path)
    assert patch.patch_id in receipt_ledger(grown)
    assert grown.tree_size_at_last_save > state.tree_size_at_last_save
    assert committed_evaluator(paths.work_dir, grown)[0] == request["candidate"]


def test_invalid_activation_completion_cannot_turn_failure_into_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed model is benign only with valid committed transition evidence."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    evaluate = worker.MorpionActiveModelNodeReevaluationEvaluator.evaluate_node_window

    def damaged(*positional: Any, **keywords: Any) -> Any:
        rows = evaluate(*positional, **keywords)
        activate_model(paths, request)
        raw = json.loads(paths.run_state_path.read_bytes())
        raw["metadata"][PUBLICATION_KEY]["generations"][-1][
            "activation_request_sha256"
        ] = "0" * 64
        paths.run_state_path.write_text(json.dumps(raw))
        return rows

    monkeypatch.setattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator,
        "evaluate_node_window",
        damaged,
    )
    with pytest.raises(ValueError):
        propose(args)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
