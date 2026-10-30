"""Cheap passive wait/lifecycle observations cannot change worker behavior."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import runtime_performance as runtime
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
    load_performance_summary,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement

if TYPE_CHECKING:
    from pathlib import Path


def test_wait_decisions_coalesce_and_keep_open_interval_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated polls retain one small open interval and close only on a new decision."""
    now = [100.0]
    monkeypatch.setattr(
        runtime,
        "time",
        SimpleNamespace(
            time=lambda: now[0],
            monotonic=lambda: now[0],
            time_ns=lambda: int(now[0] * 1e9),
        ),
    )
    runtime.observe_wait(tmp_path, "growth", "learning_boundary", 96)
    now[0] += 5
    runtime.observe_wait(tmp_path, "growth", "learning_boundary", 96)
    files = list((tmp_path / "pipeline/performance/runtime").glob("*.json"))
    assert len(files) == 1
    open_record = json.loads(files[0].read_bytes())
    assert open_record["elapsed_s"] is None
    assert open_record["observed_elapsed_s"] == 5
    summary = load_performance_summary(tmp_path)
    [row] = summary.timeline
    assert row["Elapsed (s)"] is None
    assert row["Observed incomplete wait (s)"] == 5
    assert row["Wait reason"] == "learning_boundary"
    assert not summary.diagnostics["evidence_complete"]
    now[0] += 5
    runtime.observe_wait(tmp_path, "growth", None, 96)
    [closed] = load_performance_summary(tmp_path).timeline
    assert closed["Elapsed (s)"] == 10
    assert closed["Evidence status"] == "completed"


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, ValueError])
def test_runtime_ring_is_bounded_and_failures_are_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    """Bound retained records and swallow I/O failures without changing caller control flow."""
    monkeypatch.setattr(runtime, "RUNTIME_SLOTS", 4)
    for start in range(30):
        runtime.observe_ownership(
            tmp_path,
            started_unix_s=100 + start,
            started=start,
            acquired=start + 1,
            finished=start + 3,
            outcome="released",
        )
    assert len(list((tmp_path / "pipeline/performance/runtime").glob("*.json"))) <= 4

    def fail(*args: Any, **kwargs: Any) -> None:
        message = "injected metrics failure"
        raise failure_type(message)

    monkeypatch.setattr(runtime, "_write", fail)
    runtime.observe_wait(tmp_path, "training", "claim_contention", 96)
    runtime.observe_ownership(
        tmp_path,
        started_unix_s=1,
        started=1,
        acquired=None,
        finished=2,
        outcome="timeout",
    )
    runtime.WorkerObservation(tmp_path, "training").finish(1)


def test_worker_incomplete_failed_and_cpu_scope(tmp_path: Path) -> None:
    """Distinguish process CPU, waited-for child CPU and process-lifetime RSS."""
    worker = runtime.WorkerObservation(tmp_path, "training")
    summary = load_performance_summary(tmp_path)
    assert any(
        r["Stage"] == "worker_lifetime" and r["Evidence status"] == "incomplete"
        for r in summary.timeline
    )
    worker.finish(1)
    summary = load_performance_summary(tmp_path)
    assert any(
        r["Stage"] == "worker_lifetime" and r["Evidence status"] == "failed"
        for r in summary.timeline
    )
    record = StageMeasurement().finish()
    assert record["cpu_scope"] == "current_process"
    assert (
        record["children_cpu_scope"] == "reaped_children_only; live children excluded"
    )
    assert record["rss_peak_scope"] == "process_lifetime"
    assert record["device_timing_scope"] == "host_wall; device completion unverified"


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_rejected_owned_operation_keeps_workspace_and_releases_lock(
    tmp_path: Path, failure_type: type[BaseException]
) -> None:
    """A passive ownership record cannot mutate a rejected operation's workspace."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.pipeline_dir.mkdir(parents=True)
    (paths.pipeline_dir / ".growth_patch_consumer.lock").touch()

    def inventory() -> dict[Path, bytes]:
        return {
            path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
        }

    before = inventory()
    expected = failure_type("rejected operation")
    with pytest.raises(failure_type) as caught, growth_receipt_ownership(paths):
        raise expected
    assert caught.value is expected
    assert inventory() == before
    # A fresh nonblocking acquisition succeeds, and completed scopes still report.
    with growth_receipt_ownership(paths):
        pass
    assert list((paths.pipeline_dir / "performance/runtime").glob("ownership-*.json"))
