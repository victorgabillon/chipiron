"""Passive observations preserve real activation, error and retry boundaries."""

import json
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap import performance
from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
    load_performance_summary,
)
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_model_activation_edges import propose


def sidecars(root: Path) -> list[Path]:
    """Only the new optional sidecars, not scientific/legacy artifacts."""
    return list(root.glob("pipeline/performance/generation_*/activation.phases-*.json"))


@pytest.mark.parametrize("pending", [False, True])
def test_real_phase_hierarchy_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    """Both branches retain one total/refresh; drain is conditional, never fake zero."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    if pending:
        assert propose(args).patch_written
    after = activation.activate_model(paths, request)
    [path] = sidecars(paths.work_dir)
    assert path.stat().st_size <= 65536
    data = json.loads(path.read_text())
    assert len(data["phases"]) <= 64
    assert data["dropped_rows"] == data["observer_errors"] == 0
    phases = {r["phase"]: r for r in data["phases"]}
    assert ("drain.save" in phases) == pending
    assert ("post_drain.preflight" in phases) == pending
    for role in ["activation.save", *(["drain.save"] if pending else [])]:
        assert all(
            phases[role + "." + child]["parent_phase"] == role
            for child in ("checkpoint", "export", "commit")
        )
    summary = load_performance_summary(paths.work_dir)
    rows = [r for r in summary.timeline if r["Generation"] == after.generation]
    totals = [r for r in rows if r["Stage"] == "activation"]
    refresh = [r for r in rows if r["Stage"] == "activation.refresh"]
    assert len(totals) == len(refresh) == 1
    assert (
        next(g for g in summary.generations if g["Generation"] == after.generation)[
            "activation (s)"
        ]
        == data["total"]["elapsed_s"]
    )
    children = [r for r in rows if r["Stage"].startswith("activation.phase.")]
    assert len(children) == len(phases)
    assert all(r["Clock evidence"] == "complete" for r in children)
    assert all(r["Source Dataset generation"] is None for r in children)
    assert all(r["Parent"] is None for r in children if "cleanup." in r["Stage"])
    assert all(
        r["Parent"] is not None for r in children if "cleanup." not in r["Stage"]
    )
    original = path.read_bytes()
    authority = paths.run_state_path.read_bytes()
    assert activation.activate_model(paths, request) == after
    assert paths.run_state_path.read_bytes() == authority
    assert sidecars(paths.work_dir) == [path] and path.read_bytes() == original
    # The read-only report keeps malformed clocks visible without counting them.
    data["phases"][0]["process_identity"] = "different-process"
    data["dropped_rows"] = 1
    path.write_text(json.dumps(data))
    report = load_performance_summary(paths.work_dir)
    assert any("partial activation phase" in w for w in report.warnings)
    assert any(
        r["Clock evidence"] == "inconsistent clocks"
        for r in report.timeline
        if r["Stage"].startswith("activation.phase.")
    )


def test_committed_cleanup_failure_is_not_failed_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the actual cleanup exception and distinguish committed authority."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "work")
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    close = DurableGenerationPublication.close
    error = RuntimeError("cleanup fixture")

    def fail_after_commit(self: DurableGenerationPublication) -> None:
        close(self)
        if self.state is not None and self.state.generation == 2:
            raise error

    monkeypatch.setattr(DurableGenerationPublication, "close", fail_after_commit)
    with pytest.raises(RuntimeError) as raised:
        activation.activate_model(paths, request)
    assert raised.value is error
    [path] = sidecars(paths.work_dir)
    data = json.loads(path.read_text())
    assert data["scientific_outcome"] == "committed"
    assert data["function_outcome"] == "failed"
    assert data["phases"][-1]["phase"] == "cleanup.close"
    assert data["phases"][-1]["outcome"] == "failed"
    assert data["total"] == {}  # No fabricated end for the interrupted old stage.
    report = load_performance_summary(paths.work_dir)
    assert any("total unavailable" in w for w in report.warnings)
    assert (
        next(g for g in report.generations if g["Generation"] == 2)["activation (s)"]
        is None
    )


@pytest.mark.parametrize(
    "failure", ["construct", "finish", "append", "serialize", "sink_logger"]
)
def test_observer_failures_never_mask_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Exercise the real sink's failing warning logger, not just a mock sink."""
    timing = performance.ActivationPhaseTiming({})

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        message = "observer fixture"
        raise RuntimeError(message)

    if failure == "construct":
        monkeypatch.setattr(performance, "StageMeasurement", fail)
    elif failure == "finish":
        monkeypatch.setattr(performance.StageMeasurement, "finish", fail)
    elif failure == "append":

        class BrokenList(list[dict[str, object]]):
            def append(self, value: dict[str, object]) -> None:
                fail(value)

        timing.rows = BrokenList()
    error = ValueError("scientific failure")
    with pytest.raises(ValueError) as raised, timing.phase("work"):
        raise error
    assert raised.value is error
    with timing.phase("successful"):
        answer = 42
    assert answer == 42
    if failure == "serialize":
        monkeypatch.setattr(performance.json, "dumps", fail)
    elif failure == "sink_logger":

        def no_write(*_args: Any, **_kwargs: Any) -> Any:
            message = "disk fixture"
            raise OSError(message)

        monkeypatch.setattr(Path, "write_text", no_write)
        monkeypatch.setattr(performance.LOGGER, "warning", fail)
    timing.publish(tmp_path, 2, "pub", {}, function_failed=False)


def test_bounds_and_cancellation(tmp_path: Path) -> None:
    """Bound both rows and serialized bytes; operator cancellation propagates."""
    timing = performance.ActivationPhaseTiming({})
    for i in range(70):
        with timing.phase(str(i) + "x" * 2048):
            pass
    assert len(timing.rows) == 64 and timing.dropped == 6
    with pytest.raises(KeyboardInterrupt), timing.phase("cancel"):
        raise KeyboardInterrupt
    timing.publish(tmp_path, 2, "pub", {}, function_failed=False)
    [path] = sidecars(tmp_path)
    assert path.stat().st_size <= 65536
    data = json.loads(path.read_text())
    assert data["dropped_rows"] > 7
    assert len(data["phases"]) + data["dropped_rows"] == 71


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "missing_clock",
        "bad_parent",
        "bad_total",
        "unjoined",
        "truncated",
        "malformed",
    ],
)
def test_bounded_optional_phase_reader(tmp_path: Path, fault: str) -> None:
    """Malformed optional children never manufacture coverage or another total."""
    from chipiron.environments.morpion.bootstrap.dashboard.performance_evidence import (
        EvidenceReader,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.performance_timeline import (
        interval_summary,
    )

    warnings: list[str] = []
    reader = EvidenceReader(tmp_path, warnings)
    identity = {
        "publication_id": "pub",
        "activation_request_sha256": "request",
        "attempt_id": "training",
        "training_generation": 1,
        "model_generation": 1,
        "evidence_status": "accepted",
    }
    total = {
        "started_unix_s": 100.0,
        "finished_unix_s": 120.0,
        "elapsed_s": 20.0,
        "pid": 1,
        "process_identity": "p",
    }
    reader.publications[2] = {"id": "pub", "activation_request_sha256": "request"}
    parent = reader.add(2, "activation", total, "old", identity=identity)
    reader.add(
        2,
        "activation.refresh",
        {**total, "started_unix_s": 105.0, "finished_unix_s": 109.0, "elapsed_s": 4.0},
        "descriptor",
        identity=identity,
        parent=parent,
    )
    child: dict[str, Any] = {
        **total,
        "phase": "initial.preflight",
        "ordinal": 1,
        "started_unix_s": 101.0,
        "finished_unix_s": 103.0,
        "elapsed_s": 2.0,
        "monotonic_start_s": 1.0,
        "monotonic_end_s": 3.0,
    }
    payload: dict[str, Any] = {
        **identity,
        "generation": 2,
        "phase_schema_version": 1,
        "total": dict(total),
        "phases": [child],
        "dropped_rows": 0,
        "observer_errors": 0,
    }
    if fault == "missing_clock":
        child.pop("monotonic_end_s")
    elif fault == "bad_parent":
        child["parent_phase"] = "unknown"
    elif fault == "bad_total":
        payload["total"]["elapsed_s"] = 999
    elif fault == "unjoined":
        payload["activation_request_sha256"] = "other-request"
    elif fault == "truncated":
        payload["phases"] = [{**child, "phase": str(i)} for i in range(70)]
    elif fault == "malformed":
        payload["phases"] = [None, {}]
    reader._activation_phases.append((2, "sidecar", payload))  # pylint: disable=protected-access
    reader.load_activation_phases()
    records = reader.records[2]
    assert len([r for r in records if r["stage"] == "activation"]) == 1
    assert interval_summary(records)["covered_interval_s"] == 20
    assert interval_summary(records)["max_observed_workers"] == 1
    assert len(records) <= 66
    if fault in {"missing_clock", "bad_parent"}:
        assert interval_summary(records)["missing_or_inconsistent_intervals"] == 1
    elif fault != "none":
        assert warnings
