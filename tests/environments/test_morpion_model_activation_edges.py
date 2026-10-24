"""Authority, proposal fencing and scientific rejection at model transitions."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    committed_evaluator,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

if TYPE_CHECKING:
    from pathlib import Path


def propose(args: Any) -> Any:
    """Exercise the actual bounded, Growth-scaled neural producer."""
    return worker.run_morpion_reevaluation_worker_once(
        args,
        max_nodes_per_patch=3,
        use_streaming_sharded_reader=True,
        allow_scientific_v2=True,
    )


def test_pending_m1_commits_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two complete boundaries: M1 receipt/cursor commit, then M2 activation."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    assert propose(args).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    raw = paths.pipeline_reevaluation_patch_path.read_bytes()
    after = activate_model(paths, request)
    assert after.generation == 3
    receipt = receipt_ledger(after)[patch.patch_id]
    assert receipt.committed_generation == 2
    assert (
        after.metadata["atomic_reevaluation"]["accepted_cursor"]
        == patch.metadata["atomic_reevaluation"]["next_cursor"]
    )
    assert not paths.pipeline_reevaluation_patch_path.exists()
    # Discovery is advisory; the real producer can start with M2 even if it is absent.
    paths.pipeline_active_model_path.unlink()
    assert propose(args).patch_written
    next_patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert next_patch.model_bundle_path == request["candidate"]["path"]
    assert (
        next_patch.metadata["atomic_reevaluation"]["expected_cursor_sha256"]
        != request["expected_cursor_sha256"]
    )
    assert all(row.is_exact is False for row in next_patch.rows)
    assert raw != paths.pipeline_reevaluation_patch_path.read_bytes()
    from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
    from chipiron.environments.morpion.bootstrap.model_activation import _runner

    runner, _ = _runner(paths, after)
    producer = after.metadata["durable_generation_publication_v1"]["generations"][-1][
        "scientific_identity"
    ]["producer"]["identity"]
    run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=1,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert set(receipt_ledger(accepted)) == {patch.patch_id, next_patch.patch_id}
    assert (
        accepted.metadata["atomic_reevaluation"]["accepted_cursor"]
        == next_patch.metadata["atomic_reevaluation"]["next_cursor"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "expected_head",
        "previous_model",
        "accepted_training",
        "candidate",
        "config_sha256",
        "policy",
        "expected_cursor_sha256",
        "receipt_ledger_sha256",
    ],
)
def test_stale_request_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """No preparation directory, model attachment or authority mutation on bad binding."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    bad = copy.deepcopy(request)
    bad[field] = "wrong"
    before = paths.run_state_path.read_bytes()
    with pytest.raises(GenerationPublicationError, match="Stale"):
        activate_model(paths, bad)
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "weights",
        "missing",
        "pending",
        "fallback",
        "checkpoint",
        "science",
        "no_training",
    ],
)
def test_corruption_and_unsupported_proposals_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """A failed transition cannot expose a new evaluator or consume bad work."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    before = paths.run_state_path.read_bytes()
    model = request["previous_model"]
    if damage in {"weights", "missing"}:
        file = next((paths.work_dir / request["candidate"]["path"]).glob("*.pt"))
        if damage == "missing":
            file.unlink()
        else:
            file.write_bytes(b"corrupt synthetic weights")
    elif damage == "pending":
        paths.pipeline_reevaluation_patch_path.write_text("{}")
    elif damage in {"checkpoint", "science"}:
        state = load_bootstrap_run_state(paths.run_state_path)
        target = paths.work_dir / (
            state.latest_runtime_checkpoint_path
            if damage == "checkpoint"
            else state.latest_tree_snapshot_path
        )
        if target.is_dir():
            target = target / "manifest.json"
        target.write_bytes(b"deliberately corrupted synthetic dependency")
    elif damage == "no_training":
        (paths.work_dir / request["candidate"]["path"]).parents[2].joinpath(
            "proposal.json"
        ).unlink()
    with pytest.raises((ValueError, KeyError, TypeError, OSError)):
        if damage == "fallback":
            worker.run_morpion_reevaluation_worker_once(
                args, use_snapshot_value_fallback=True
            )
        else:
            activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before
    if damage in {"checkpoint", "science"}:
        with pytest.raises(ValueError):
            committed_evaluator(
                paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
            )
        return
    assert (
        committed_evaluator(
            paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
        )[0]
        == model
    )


def test_second_activation_and_ordinary_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1→M2→M3 uses existing Training and Growth; retry cannot reactivate M2."""
    from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
    from chipiron.environments.morpion.bootstrap.model_activation import _runner
    from chipiron.environments.morpion.bootstrap.pipeline.stages import (
        run_pipeline_training_stage,
    )
    from tests.environments.morpion_fenced_dataset_helpers import run_dataset

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, initial = setup_activation(tmp_path / "workspace")
    first = prepare_model_activation_request(paths, training_generation=1)
    second_head = activate_model(paths, first)
    run_dataset(args, generation=second_head.generation)
    run_pipeline_training_stage(args, generation=second_head.generation)
    second = prepare_model_activation_request(paths, training_generation=2)
    third_head = activate_model(paths, second)
    assert second["candidate"]["sha256"] != first["candidate"]["sha256"]
    assert activate_model(paths, first) == third_head
    # Run the actual stage with a stale discovery projection and the original producer.
    paths.pipeline_active_model_path.unlink()
    runner, _ = _runner(paths, third_head)
    initial_producer = initial.metadata["durable_generation_publication_v1"][
        "generations"
    ][0]["scientific_identity"]["producer"]["identity"]
    run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=1,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=initial_producer,
        closed_scientific_snapshot=True,
    )
    grown = load_bootstrap_run_state(paths.run_state_path)
    assert grown.generation == 4
    assert grown.tree_size_at_last_save > third_head.tree_size_at_last_save
    assert committed_evaluator(paths.work_dir, grown)[0] == second["candidate"]


def test_exact_receipt_retry_after_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old exact bytes acknowledge without replay; a reused UUID with new bytes refuses."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    assert propose(args).patch_written
    raw = paths.pipeline_reevaluation_patch_path.read_bytes()
    after = activate_model(paths, request)
    before = paths.run_state_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.write_bytes(raw)
    assert activate_model(paths, request) == after
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert paths.run_state_path.read_bytes() == before
    paths.pipeline_reevaluation_patch_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="Conflicting reused patch ID"):
        activate_model(paths, request)
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == raw + b" "
    assert paths.run_state_path.read_bytes() == before


def test_growth_lock_excludes_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing flock alone excludes competing Growth and activation."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        growth_receipt_ownership,
    )

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    before = paths.run_state_path.read_bytes()
    with growth_receipt_ownership(paths), pytest.raises(BlockingIOError):
        activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before


def test_late_m1_producer_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pause real M1 computation, commit M2, then exercise the final producer fence."""
    from threading import Event, Thread

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    arrived, release = Event(), Event()
    original = worker.proposal_patch
    result: dict[str, Any] = {}

    def pause(*args: Any, **kwargs: Any) -> Any:
        patch = original(*args, **kwargs)
        arrived.set()
        assert release.wait(30)
        return patch

    def produce() -> None:
        try:
            result["value"] = propose(args)
        except Exception as exc:
            result["error"] = exc

    monkeypatch.setattr(worker, "proposal_patch", pause)
    thread = Thread(target=produce)
    thread.start()
    try:
        assert arrived.wait(30), result
        after = activate_model(paths, request)
    finally:
        release.set()
        thread.join(30)
    assert not thread.is_alive()
    assert "Model changed" in str(result.get("error")), result
    assert load_bootstrap_run_state(paths.run_state_path) == after
    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_competing_activation_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real owner excludes a competing request; retry recognizes the same commit."""
    from threading import Event, Thread

    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
    )

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    arrived, release = Event(), Event()
    original = AnemoneMorpionSearchRunner.refresh_model_estimates
    results: dict[str, Any] = {}

    def pause(*args: Any, **kwargs: Any) -> Any:
        arrived.set()
        assert release.wait(30)
        return original(*args, **kwargs)

    def activate() -> None:
        try:
            results["state"] = activate_model(paths, request)
        except Exception as exc:
            results["error"] = exc

    monkeypatch.setattr(AnemoneMorpionSearchRunner, "refresh_model_estimates", pause)
    thread = Thread(target=activate)
    thread.start()
    try:
        assert arrived.wait(30), results
        with pytest.raises(BlockingIOError):
            activate_model(paths, copy.deepcopy(request))
    finally:
        release.set()
        thread.join(30)
    assert not thread.is_alive() and "error" not in results, results
    assert activate_model(paths, request) == results["state"]
