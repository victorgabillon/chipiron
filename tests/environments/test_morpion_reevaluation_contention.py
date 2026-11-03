"""Exercise real publication contention across two persistent Reevaluation jobs."""

import json
import os
from contextlib import contextmanager
from importlib import import_module
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import persistent_worker as persistent
from chipiron.environments.morpion.bootstrap import persistent_worker_state as status
from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GrowthOwnershipBlockingIOError,
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
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_model_activation_edges import propose
from tests.environments.test_morpion_persistent_workers import _identity, _launcher

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_activation_outlasts_wait_then_persistent_job_reloads_model_and_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold a real activation flock past a short wait budget; never publish M1 work.

    Dispatcher ancestry/configuration and cleanup hooks are fixture controls;
    activation, inference, kernel contention, next-job publication and Growth are real.
    """
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    authority_before = paths.run_state_path.read_bytes()
    held, release = Event(), Event()
    activation: dict[str, Any] = {}
    results = []
    cleanups = []
    refresh = AnemoneMorpionSearchRunner.refresh_model_estimates
    proposal = worker.proposal_patch

    def pause_refresh(*positional: Any, **keywords: Any) -> Any:
        held.set()
        assert release.wait(30), "Activation was not released"
        return refresh(*positional, **keywords)

    def activate() -> None:
        try:
            activation["state"] = activate_model(paths, request)
        except Exception as exc:
            activation["error"] = exc

    thread = Thread(target=activate)

    def overlap(*positional: Any, **keywords: Any) -> Any:
        patch = proposal(*positional, **keywords)
        if not held.is_set():
            thread.start()
            assert held.wait(30), activation
        return patch

    @contextmanager
    def execution(configured: Any) -> Iterator[Any]:
        yield configured

    def job(_configured: Any, *, _job_id: int | None = None) -> int:
        assert _job_id == len(results) + 1
        results.append(propose(args))
        return 0

    def cleanup() -> None:
        cleanups.append(len(results))
        if len(results) == 1:
            assert not paths.pipeline_reevaluation_patch_path.exists()
            assert not paths.pipeline_reevaluation_cursor_path.exists()
            assert paths.run_state_path.read_bytes() == authority_before
            release.set()
            thread.join(30)
            assert not thread.is_alive() and "error" not in activation, activation

    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "0.05")
    monkeypatch.setattr(
        AnemoneMorpionSearchRunner, "refresh_model_estimates", pause_refresh
    )
    monkeypatch.setattr(worker, "proposal_patch", overlap)
    monkeypatch.setattr(
        persistent,
        "monitored_identity",
        lambda: {**_identity(), "session": os.getpid()},
    )
    monkeypatch.setattr(status, "monitored_identity", _identity)
    monkeypatch.setattr(persistent, "launcher_execution", execution)
    # Full-suite package import checks can replace the launcher module in
    # sys.modules after this test file was collected. Persistent dispatch imports
    # the current module at job time, so patch that same module object.
    current_launcher = import_module("chipiron.environments.morpion.bootstrap.launcher")
    monkeypatch.setattr(current_launcher, "_run_launcher_cli", job)
    monkeypatch.setattr(persistent, "_cleanup", cleanup)
    try:
        assert (
            persistent.run_persistent_worker(
                _launcher(paths.work_dir, pipeline_stage="reevaluation")
            )
            == 0
        )
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(30)
            assert not thread.is_alive()
    assert [result.reason for result in results] == ["ownership_contended", None]
    assert cleanups == [1, 2]
    process = json.loads(
        (paths.pipeline_dir / "worker_processes/reevaluation.json").read_text()
    )
    assert process["phase"] == "stopped" and process["exit_code"] == 0
    assert process["completed_jobs"] == 2
    records = [
        json.loads(p.read_text())
        for p in (paths.pipeline_dir / "performance").rglob("*.json")
    ]
    assert any(r.get("outcome") == "contention_timeout" for r in records)
    assert any(
        r.get("reason") == "ownership_contended" and r.get("patch_written") is False
        for r in records
    )
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert patch.metadata["atomic_reevaluation"]["model"] == request["candidate"]
    state = activation["state"]
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


@pytest.mark.parametrize("boundary", ["inference", "publication"])
@pytest.mark.parametrize(
    "error_type", [BlockingIOError, GrowthOwnershipBlockingIOError]
)
def test_io_failure_outside_lock_acquisition_is_still_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    error_type: type[BlockingIOError],
) -> None:
    """Only entering the ownership context may defer; computation/body failures escape."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    before = paths.run_state_path.read_bytes()

    def fail(*_positional: Any, **_keywords: Any) -> Any:
        message = "injected non-acquisition failure"
        raise error_type(message)

    target, name = (
        (worker.MorpionActiveModelNodeReevaluationEvaluator, "evaluate_node_window")
        if boundary == "inference"
        else (worker, "_save_reevaluation_patch_exclusive")
    )
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(error_type, match="injected non-acquisition failure"):
        propose(args)
    assert paths.run_state_path.read_bytes() == before
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
