"""Collection alone must never count as complete CI execution."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def metrics() -> ModuleType:
    """Load the existing standalone reporter without changing import paths."""
    path = Path(__file__).resolve().parents[2] / "scripts/ci_test_metrics.py"
    spec = importlib.util.spec_from_file_location("metrics_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.fast
@pytest.mark.parametrize(
    "fault",
    [
        "missing_setup",
        "duplicate_setup",
        "missing_call",
        "extra_call",
        "missing_teardown",
        "duplicate_identity",
        "unexpected_test",
    ],
)
def test_execution_guard_refuses_incomplete_or_duplicate_evidence(
    metrics: ModuleType, fault: str
) -> None:
    """Missing work and repeated work both invalidate a purported full pass."""
    recorder = metrics.TestMetrics()
    recorder.selected = ["test_ok", "test_skip"]
    recorder.reports = [
        {"nodeid": "test_ok", "phase": phase, "outcome": "passed"}
        for phase in ("setup", "call", "teardown")
    ] + [
        {"nodeid": "test_skip", "phase": "setup", "outcome": "skipped"},
        {"nodeid": "test_skip", "phase": "teardown", "outcome": "passed"},
    ]
    assert recorder.execution_errors() == []
    if fault.startswith("missing_"):
        phase = fault.removeprefix("missing_")
        recorder.reports = [r for r in recorder.reports if r["phase"] != phase]
    elif fault == "duplicate_setup":
        recorder.reports.append(recorder.reports[0])
    elif fault == "extra_call":
        recorder.reports.append({
            "nodeid": "test_skip",
            "phase": "call",
            "outcome": "passed",
        })
    elif fault == "duplicate_identity":
        recorder.selected.append("test_ok")
    else:
        recorder.reports.append({
            "nodeid": "unselected",
            "phase": "setup",
            "outcome": "passed",
        })
    assert recorder.execution_errors()


@pytest.mark.fast
@pytest.mark.parametrize("collect_only", [False, True])
def test_reporter_preserves_real_execution_and_collection(
    tmp_path: Path, collect_only: bool
) -> None:
    """Real passes/skips and intentional collection-only runs retain distinct evidence."""
    (tmp_path / "test_probe.py").write_text(
        "import pytest\ndef test_ok(): pass\n"
        "@pytest.mark.skip(reason='probe skip')\ndef test_skip(): pass\n"
    )
    script = Path(__file__).resolve().parents[2] / "scripts/ci_test_metrics.py"
    args = [sys.executable, str(script), "-q", "test_probe.py"]
    if collect_only:
        args.append("--collect-only")
    result = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / ".ci-reports/tests.json").read_text())
    assert len(report["selected_ids"]) == 2
    assert report["collect_only"] is collect_only
    assert report["execution_errors"] == []
    assert bool(report["reports"]) is not collect_only


@pytest.mark.fast
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("fails", [False, True])
def test_module_plugin_preserves_phases_and_exit_status(
    tmp_path: Path, workers: int, fails: bool
) -> None:
    """The integration observer records real distributed passes, skips and failures."""
    (tmp_path / "test_probe.py").write_text(
        f"import pytest\ndef test_result(): assert {not fails}\n"
        "@pytest.mark.skip(reason='probe skip')\ndef test_skip(): pass\n"
    )
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "integration.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYTEST_PLUGINS": "scripts.ci_test_metrics",
        "CI_TEST_METRICS_OUTPUT": str(output),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", str(workers), "test_probe.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == int(fails), result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert len(report["selected_ids"]) == 2
    assert report["execution_errors"] == []
    assert report["exit_code"] == int(fails)
    assert len(report["reports"]) == 5
    assert len(report["worker_collections"]) == workers
