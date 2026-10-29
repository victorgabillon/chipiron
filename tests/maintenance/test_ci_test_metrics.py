"""Collection alone must never count as complete CI execution."""

import importlib.util
import json
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
