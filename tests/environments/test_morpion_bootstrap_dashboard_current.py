"""K1 regressions for bounded current-pipeline monitoring and honest provenance."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap.dashboard import runtime_view as rv
from chipiron.environments.morpion.bootstrap.dashboard import view_model as vm
from chipiron.environments.morpion.bootstrap.dashboard.pipeline_view import (
    resource_summary,
    schedule_summary,
    score_rows,
)


def put(path: Path, data: Any) -> None:
    """Write only isolated fixture artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    """Completed early goal with normal SIGINT shutdown of the canonical monitor."""
    root = tmp_path / "runtime"
    work = root / "work"
    command = ["python", "--work-dir", str(work)]
    owner = {"pid": 999999999, "start_ticks": "1", "boot_id": "old"}
    put(
        root / "LAUNCH_READY.json",
        {
            "runtime": str(root),
            "scientific_work_dir": str(work),
            "commands": {"run": command},
        },
    )
    put(
        root / "status.json",
        {
            "phase": "completed",
            "reason": "early_goal",
            "updated_utc": "2026-01-01T00:01:00Z",
            "errors": [],
        },
    )
    put(root / "owner.json", owner)
    put(
        root / "reports/lease-release.json",
        {"owner": owner, "children": [], "released_utc": "2026-01-01T00:01:01Z"},
    )
    put(
        root / "reports/run-receipt.json",
        {
            "commands": {"run": command},
            "started_utc": "2026-01-01T00:00:00Z",
            "reason": "early_goal",
            "owner_elapsed_s": 60,
            "errors": [],
            "last_observation": {"head": "head", "early_goal": True},
            "ownership": {
                "confirmed": True,
                "survivors": [],
                "remaining_children": [],
                "post_helpers_reaped": True,
            },
        },
    )
    put(
        work / "bootstrap_config.json",
        {
            "pipeline_mode": "artifact_pipeline",
            "metadata": {"learning_schedule": {"growth_lookahead_cycles": 4}},
        },
    )
    put(
        work / "run_state.json",
        {
            "generation": 14,
            "cycle_index": 10,
            "metadata": {
                "durable_generation_publication_v1": {"head_id": "head"},
                "learning_window_v1": {
                    "active": {
                        "source_cycle": 8,
                        "target_cycle": 12,
                        "source": {"generation": 10, "id": "reserved"},
                    }
                },
            },
        },
    )
    put(
        work / "launcher_process_state.json", {"stopped_at_utc": "2026-01-01T00:00:59Z"}
    )
    put(work / "resource_monitor/usage.json", {"session": "123"})
    put(
        work / "resource_monitor/session_123.json",
        {
            "shutdown_confirmed": True,
            "supervisor_exit_code": 143,
            "reason": "operator_SIGINT",
        },
    )
    return work, root


def change(path: Path, **values: Any) -> None:
    """Change one fixture receipt without replacing independent evidence."""
    data = json.loads(path.read_text())
    data.update(values)
    put(path, data)


def test_goal_and_explicit_association(run: tuple[Path, Path]) -> None:
    """Parent status alone is ignored; associated goal and cleanup override exit143."""
    work, root = run
    assert vm.load_dashboard_snapshot(work).status == "stopped"
    snapshot = vm.load_dashboard_snapshot(work, root)
    assert snapshot.status == "completed"
    assert "early_goal" in snapshot.status_detail
    assert {w.state for w in snapshot.workers} == {"stopped"}
    change(root / "LAUNCH_READY.json", scientific_work_dir="/unrelated")
    snapshot = vm.load_dashboard_snapshot(work, root)
    assert snapshot.status == "stopped" and "association refused" in snapshot.errors[0]


@pytest.mark.parametrize(
    ("reason", "phase", "expected"),
    [
        ("operator_cancel", "stopped", "stopped"),
        ("runtime_failure", "failed", "failed"),
        ("early_goal", "completed", "completed"),
    ],
)
def test_outcomes(
    run: tuple[Path, Path], reason: str, phase: str, expected: str
) -> None:
    """Intentional stop, failure and successful declared goal stay distinct."""
    work, root = run
    change(root / "reports/run-receipt.json", reason=reason)
    change(root / "status.json", reason=reason, phase=phase)
    assert vm.load_dashboard_snapshot(work, root).status == expected


@pytest.mark.parametrize(
    "damage", ["cleanup", "release", "head", "version", "partial", "time", "errors"]
)
def test_incomplete_or_contradictory_evidence(
    run: tuple[Path, Path], damage: str
) -> None:
    """No one success bit may erase cleanup failures, corruption or stale evidence."""
    work, root = run
    receipt = root / "reports/run-receipt.json"
    if damage == "cleanup":
        change(receipt, ownership={"confirmed": False})
    elif damage == "release":
        (root / "reports/lease-release.json").unlink()
    elif damage == "head":
        change(receipt, last_observation={"head": "previous", "early_goal": True})
    elif damage == "version":
        change(receipt, schema_version=99)
    elif damage == "partial":
        receipt.write_text("{")
    elif damage == "time":
        change(root / "status.json", updated_utc="2025-01-01T00:00:00Z")
    else:
        change(receipt, errors=["worker failure"])
    assert vm.load_dashboard_snapshot(work, root).status != "completed"


def test_live_pid_requires_boot_and_start_ticks(tmp_path: Path) -> None:
    """A reused PID or another process command cannot create a live run."""
    ticks = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]
    identity = {
        "pid": os.getpid(),
        "start_ticks": ticks,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }
    assert rv.live_identity(identity)
    assert not rv.live_identity({**identity, "start_ticks": "0"})
    assert not rv.live_identity({**identity, "boot_id": "previous-boot"})
    assert not rv.live_identity(identity, tmp_path)


@pytest.mark.parametrize(
    ("phase", "age", "schema", "expected"),
    [
        ("idle", 1, 1, "idle"),
        ("idle", 10, 1, "unknown"),
        ("running", 100, 1, "working"),
        ("cleanup", 100, 1, "working"),
        ("idle", 1, 99, "unknown"),
    ],
)
def test_persistent_job_state(
    run: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    age: int,
    schema: int,
    expected: str,
) -> None:
    """Idle live processes differ from active jobs; active boundary heartbeat may be old."""
    work, _ = run
    monkeypatch.setattr(rv, "live_identity", lambda *args: True)
    put(
        work / "pipeline/worker_processes/growth.json",
        {
            "schema_version": schema,
            "worker": "growth",
            "identity": {"pid_namespace": os.readlink("/proc/self/ns/pid")},
            "phase": phase,
            "job_id": 7,
            "heartbeat_monotonic_s": time.monotonic() - age,
            "job_started_monotonic_s": time.monotonic() - 200,
            "updated_unix_s": time.time(),
        },
    )
    snapshot = vm.load_dashboard_snapshot(work)
    assert snapshot.workers[0].state == expected
    if expected == "working":
        assert "Active job age" in snapshot.workers[0].detail


def test_current_metadata_precedence_and_loss_name(run: tuple[Path, Path]) -> None:
    """Flat active-bundle fields win over legacy fields, never latest training metrics."""
    work, _ = run
    put(
        work / "pipeline/active_model.json",
        {"evaluator_name": "active", "model_bundle_path": "models/active"},
    )
    put(
        work / "models/active/morpion_manifest.json",
        {
            "metadata": {
                "num_epochs": 5,
                "num_train_samples": 100,
                "num_validation_samples": 20,
                "validation_loss": 3.5,
                "loss_name": "mse",
                "training_config": {"config": {"num_epochs": 99}},
            }
        },
    )
    summary = vm.active_model_summary(vm.load_dashboard_snapshot(work))
    assert summary["Epochs"] == 5 and summary["Training rows"] == 100
    assert summary["Validation rows"] == 20 and summary["Validation MSE"] == 3.5
    snapshot = vm.load_dashboard_snapshot(work)
    snapshot.model_manifest["metadata"]["loss_name"] = "mae"
    assert vm.active_model_summary(snapshot)["Validation MSE"] is None


def test_units_selection_age_and_missing_resources(run: tuple[Path, Path]) -> None:
    """Selected/accepted Training is not activation, and lag uses publication generations."""
    work, root = run
    snapshot = replace(
        vm.load_dashboard_snapshot(work, root),
        active_model={
            "evaluator_name": "old",
            "source_generation": 4,
            "source": "local_training",
        },
        dataset={"generation": 10},
        training={"generation": 10, "selected_evaluator_name": "candidate"},
    )
    summary = schedule_summary(snapshot)
    assert summary["Completed ordinary cycles in window"] == 2
    assert summary["Remaining ordinary cycles in window"] == 2
    assert summary["Dataset lag · publication generations"] == 4
    assert summary["Model lag · publication generations"] == 10
    assert summary["Selected candidate · latest Training"] == "candidate"
    assert summary["Actually active model"] == "old"
    assert summary["Pending activation"] is None
    resources = resource_summary(snapshot)
    assert resources["GPU utilization"] is None
    assert resources["Reported active-bundle profile"] is None


def test_certified_score_time_and_coverage(run: tuple[Path, Path]) -> None:
    """36 initial points are not moves; Dataset publication is not discovery time."""
    work, root = run
    snapshot = replace(
        vm.load_dashboard_snapshot(work, root),
        dataset_history=(
            {
                "updated_at_utc": "2026-01-01T00:00:20Z",
                "record_status": {
                    "current_best_total_points": 100,
                    "current_best_moves_since_start": 64,
                    "current_best_is_terminal": True,
                },
            },
            {
                "updated_at_utc": "2026-01-01T00:00:30Z",
                "record_status": {"current_best_total_points": 200},
            },
        ),
    )
    rows = score_rows(snapshot)
    assert len(rows) == 1 and rows[0]["value"] == 100 and rows[0]["time"] == 20
    assert rows[0]["origin"] == "Dataset publication"
    assert score_rows(replace(snapshot, runtime={})) == []


def test_bounded_observation_and_no_mutation(
    run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No huge reports, tree/weights/GPU imports, or writes during repeated refresh."""
    import builtins

    work, root = run
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith((
            "torch",
            "chipiron.environments.morpion.bootstrap.runtime",
            "chipiron.environments.morpion.bootstrap.dashboard.tree_inspector",
        ))
        return original(name, *args, **kwargs)

    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(builtins, "__import__", guarded)
    vm.load_dashboard_snapshot(work, root)
    vm.load_dashboard_snapshot(work, root)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    path = work / "too-big.json"
    path.write_bytes(b" " * 100)
    errors: list[str] = []
    assert vm.read_artifact(path, errors, limit=10) == {} and "exceeds" in errors[0]


def test_pending_activation_requires_request_evidence(run: tuple[Path, Path]) -> None:
    """A selected candidate alone is not a pending request or a committed activation."""
    import hashlib

    work, root = run
    request = {
        "schema_version": 1,
        "training_generation": 10,
        "candidate": {"name": "next"},
    }
    snapshot = replace(
        vm.load_dashboard_snapshot(work, root), activation_requests=(request,)
    )
    assert "no committed completion" in schedule_summary(snapshot)["Pending activation"]
    digest = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot.run_state["metadata"]["durable_generation_publication_v1"][
        "generations"
    ] = [{"generation": 14, "activation_request_sha256": digest}]
    assert "has a committed" in schedule_summary(snapshot)["Pending activation"]


def test_legacy_wait_uses_verified_identity_and_freshness(
    run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampled learning-gate wait is current only while its identity remains live."""
    work, _ = run
    put(
        work / "pipeline/performance/runtime/wait-current-growth-1.json",
        {
            "schema_version": 1,
            "stage": "wait",
            "worker": "growth",
            "reason": "await_learning_resolution",
            "generation": 10,
            "stream_id": "growth:123:456",
            "boot_id": "boot",
            "last_observed_unix_s": time.time(),
        },
    )
    monkeypatch.setattr(rv, "live_identity", lambda *args: True)
    worker = vm.load_dashboard_snapshot(work).workers[0]
    assert worker.state == "waiting" and "learning gate" in worker.detail
    monkeypatch.setattr(rv, "live_identity", lambda *args: False)
    assert vm.load_dashboard_snapshot(work).workers[0].state == "unknown"


def test_populated_pages_read_only(run: tuple[Path, Path]) -> None:
    """Exercise real Overview/Operations/Evaluator rendering with nine candidate bundles."""
    from streamlit.testing.v1 import AppTest

    work, root = run
    candidates = {}
    for i in range(9):
        name = f"candidate_{i}"
        bundle = f"models/{name}"
        candidates[name] = {"model_bundle_path": bundle, "validation_loss": i + 0.5}
        put(
            work / bundle / "morpion_manifest.json",
            {
                "metadata": {
                    "num_epochs": 5,
                    "num_train_samples": 100,
                    "continuation": {
                        "parent_bundle_sha256": f"parent-{i}",
                        "loaded_previous_weights": True,
                    },
                }
            },
        )
    put(
        work / "pipeline/generation_000010/training_status.json",
        {
            "generation": 10,
            "evaluator_results": candidates,
            "selected_evaluator_name": "candidate_8",
        },
    )
    put(
        work / "pipeline/active_model.json",
        {
            "evaluator_name": "candidate_0",
            "model_bundle_path": "models/candidate_0",
            "source_generation": 4,
        },
    )
    put(
        work / "pipeline/generation_000010/dataset_status.json",
        {
            "generation": 10,
            "updated_at_utc": "2026-01-01T00:00:20Z",
            "record_status": {
                "current_best_total_points": 100,
                "current_best_is_terminal": True,
            },
        },
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    app = AppTest.from_string(
        "from pathlib import Path\nfrom chipiron.environments.morpion.bootstrap.dashboard.app import run_dashboard_app\n"
        f"run_dashboard_app(Path({str(work)!r}), Path({str(root)!r}))",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any("versus elapsed seconds" in str(m.value) for m in app.markdown)
    for page in ("Operations", "Evaluator", "Overview"):
        app.radio(key="bootstrap_operator_page").set_value(page).run()
        assert not app.exception
        assert not {"Start", "Stop", "Restart", "Apply changes"}.intersection(
            b.label for b in app.button
        )
        if page == "Evaluator":
            assert any(len(frame.value) == 9 for frame in app.dataframe)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_closed_snapshot_heavy_history_gap_is_explicit(tmp_path: Path) -> None:
    """Validate current closed science with its reader and document legacy history's gap."""
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        _load_resolved_training_tree_snapshot,
        _ResolvedTreeSnapshotReference,
    )
    from tests.environments.morpion_durable_closed_helpers import setup_closed_baseline

    _, paths, state = setup_closed_baseline(tmp_path)
    source = paths.work_dir / state.latest_tree_snapshot_path
    assert len(load_closed_scientific_snapshot(source).snapshot.nodes) == 2
    # Full history still uses the legacy flat/sharded materializer. K1 must not
    # silently invoke the full closed-science oracle on an overview refresh.
    assert (
        _load_resolved_training_tree_snapshot(
            _ResolvedTreeSnapshotReference(
                snapshot_path=source, snapshot_source="fixture"
            )
        )
        is None
    )


def test_malformed_fields_and_live_start(run: tuple[Path, Path]) -> None:
    """Malformed valid JSON is unavailable evidence; live owner creation can anchor time."""
    work, root = run
    change(root / "status.json", phase=[])
    assert vm.load_dashboard_snapshot(work, root).status == "unknown"
    snapshot = replace(
        vm.load_dashboard_snapshot(work),
        runtime={"owner": {"created": 1767225600}},
        dataset_history=(
            {
                "updated_at_utc": "2026-01-01T00:00:20Z",
                "record_status": {
                    "current_best_total_points": 100,
                    "current_best_is_exact": True,
                },
            },
        ),
    )
    row = score_rows(snapshot)[0]
    assert row["time"] == 20 and "process creation" in row["elapsed_origin"]
    errors: list[str] = []
    path = work / "nan.json"
    path.write_text('{"value":NaN}')
    assert vm.read_artifact(path, errors) == {} and errors


@pytest.mark.parametrize(("code", "expected"), [(0, "stopped"), (1, "failed")])
def test_saved_persistent_exit(
    run: tuple[Path, Path], code: int, expected: str
) -> None:
    """An explicit stopped-phase exit is historical evidence, never process liveness."""
    work, _ = run
    put(
        work / "pipeline/worker_processes/growth.json",
        {
            "schema_version": 1,
            "worker": "growth",
            "phase": "stopped",
            "identity": {"pid": 999999999},
            "exit_code": code,
            "updated_unix_s": time.time(),
            "job_id": 7,
        },
    )
    worker = vm.load_dashboard_snapshot(work).workers[0]
    assert worker.state == expected and "no live-progress claim" in worker.detail


def test_goal_shutdown_contradiction(run: tuple[Path, Path]) -> None:
    """Conflicting cleanup receipts cannot become a successful completed run."""
    work, root = run
    change(work / "resource_monitor/session_123.json", shutdown_confirmed=False)
    snapshot = vm.load_dashboard_snapshot(work, root)
    assert snapshot.status == "unknown" and "Contradictory" in snapshot.status_detail


def test_live_runtime_owner_must_belong_to_selected_root(
    run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a verified live PID cannot supply liveness for a different runtime root."""
    work, root = run
    (root / "reports/run-receipt.json").unlink()
    (root / "reports/lease-release.json").unlink()
    change(root / "status.json", phase="running")
    change(
        root / "owner.json", command=["python", "/another/scripts/controller.py", "run"]
    )
    monkeypatch.setattr(rv, "live_identity", lambda *args: True)
    assert vm.load_dashboard_snapshot(work, root).status == "unknown"
    change(
        root / "owner.json",
        command=["python", str(root / "scripts/controller.py"), "run"],
    )
    assert vm.load_dashboard_snapshot(work, root).status == "running"
