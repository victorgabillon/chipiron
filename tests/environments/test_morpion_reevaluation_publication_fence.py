"""Deterministic producer races at the existing Growth flock boundary."""

from dataclasses import replace
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_active_model,
    save_pipeline_active_model,
)
from tests.environments.morpion_atomic_reevaluation_helpers import (
    atomic_worker,
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def paused_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Pause the actual worker before publication, without a timing-based sleep."""
    args, paths, initial = setup_atomic(tmp_path)
    arrived, release = Event(), Event()
    results: dict[str, Any] = {}
    proposal = worker.proposal_patch

    def pause(*args: Any, **kwargs: Any) -> Any:
        result = proposal(*args, **kwargs)
        arrived.set()
        assert release.wait(30), "Parent did not release publication"
        return result

    monkeypatch.setattr(worker, "proposal_patch", pause)

    def produce() -> None:
        try:
            results["result"] = atomic_worker(tmp_path, monkeypatch)
        except Exception as exc:  # surfaced and asserted in the parent test
            results["error"] = exc

    thread = Thread(target=produce)
    thread.start()
    try:
        assert arrived.wait(30), results
        yield args, paths, initial, results, release, thread
    finally:
        release.set()
        thread.join(30)
        assert not thread.is_alive(), "Producer failed to stop"


@pytest.mark.parametrize("change", ["model", "cursor"])
def test_changed_identity_cannot_publish_old_proposal(
    paused_producer: Any, change: str
) -> None:
    """Recheck identity after inference, inside the same lease as publication."""
    args, paths, _initial, results, release, thread = paused_producer
    if change == "model":
        # There is no model activation transaction yet. Simulate only its identity
        # visibility under the REAL shared lease, using a fresh synthetic workspace.
        with growth_receipt_ownership(paths):
            active = load_pipeline_active_model(paths.pipeline_active_model_path)
            save_pipeline_active_model(
                replace(active, evaluator_name="simulated-next-model"),
                paths.pipeline_active_model_path,
            )
    else:
        # A real accepted M1 patch changes the authoritative cursor while this
        # producer still holds its old expected-cursor digest.
        publish_proposal(paths, identifier="other-proposal")
        run_closed_publication(args, ClosedPublicationRunner())
    release.set()
    thread.join(30)
    assert not thread.is_alive()
    assert "error" in results, results
    assert (
        "Model changed" in str(results["error"])
        if change == "model"
        else ("Stale expected accepted cursor" in str(results["error"]))
    )
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_publication_excludes_concurrent_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock covers create-only visibility itself, not just its preceding check."""
    _args, paths, _initial = setup_atomic(tmp_path)
    arrived, release = Event(), Event()
    results: dict[str, Any] = {}
    publish = worker._save_reevaluation_patch_exclusive

    def pause(*args: Any, **kwargs: Any) -> Any:
        arrived.set()
        assert release.wait(30), "Parent did not release publication"
        return publish(*args, **kwargs)

    monkeypatch.setattr(worker, "_save_reevaluation_patch_exclusive", pause)

    def produce() -> None:
        try:
            results["result"] = atomic_worker(tmp_path, monkeypatch)
        except Exception as exc:
            results["error"] = exc

    thread = Thread(target=produce)
    thread.start()
    try:
        assert arrived.wait(30), results
        with pytest.raises(BlockingIOError), growth_receipt_ownership(paths):
            pytest.fail("Growth entered during proposal publication")
    finally:
        release.set()
        thread.join(30)
    assert not thread.is_alive()
    assert "error" not in results, results
    assert results["result"].patch_written
    with growth_receipt_ownership(paths):
        assert paths.pipeline_reevaluation_patch_path.exists()


def test_busy_growth_refuses_without_cursor_or_patch_and_retry_succeeds(
    paused_producer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep existing nonblocking flock semantics; busy publication advances nothing."""
    _args, paths, initial, results, release, thread = paused_producer
    with growth_receipt_ownership(paths):
        release.set()
        thread.join(30)
        assert not thread.is_alive()
        assert isinstance(results.get("error"), BlockingIOError), results
        assert not paths.pipeline_reevaluation_patch_path.exists()
        assert load_bootstrap_run_state(paths.run_state_path) == initial
    assert atomic_worker(paths.work_dir, monkeypatch).patch_written


def test_unrelated_growth_status_does_not_invalidate_proposal(
    paused_producer: Any,
) -> None:
    """An unrelated status update is not a source/model/cursor conflict."""
    args, paths, initial, results, release, thread = paused_producer
    run_closed_publication(args, ClosedPublicationRunner())
    release.set()
    thread.join(30)
    assert not thread.is_alive()
    assert "error" not in results, results
    assert results["result"].patch_written
    assert results["result"].tree_generation == 1
    assert (
        load_bootstrap_run_state(paths.run_state_path).cycle_index > initial.cycle_index
    )


def test_inference_does_not_hold_growth_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing kernel lease remains available during expensive computation."""
    _args, paths, _initial = setup_atomic(tmp_path)
    evaluate = worker.MorpionActiveModelNodeReevaluationEvaluator.evaluate_node_window
    observed = []

    def checked(self: Any, nodes: Any) -> Any:
        with growth_receipt_ownership(paths):
            observed.append(True)
        return evaluate(self, nodes)

    monkeypatch.setattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator,
        "evaluate_node_window",
        checked,
    )
    assert atomic_worker(tmp_path, monkeypatch).patch_written
    assert observed == [True]
