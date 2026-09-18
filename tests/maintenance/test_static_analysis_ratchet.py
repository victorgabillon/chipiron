"""The analyzer ratchet detects new debt without hiding inherited diagnostics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def ratchet() -> ModuleType:
    """Load the standalone checker without adding repository source paths."""
    path = Path(__file__).resolve().parents[2] / "scripts/static_analysis_ratchet.py"
    spec = importlib.util.spec_from_file_location("static_ratchet", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnostic(ratchet: ModuleType, root: Path, *, line: int = 2) -> dict[str, object]:
    """Parse a representative real mypy JSON diagnostic."""
    raw = json.dumps({
        "file": str(root / "module.py"),
        "line": line,
        "message": 'Incompatible types in assignment (expression has type "str", variable has type "int")',
        "code": "assignment",
        "severity": "error",
    })
    return ratchet.parse_diagnostics("mypy", raw, root)[0]


def test_line_shifts_pass_but_new_scopes_and_duplicate_debt_fail(
    ratchet: ModuleType, tmp_path: Path
) -> None:
    """Stable locations tolerate inserted lines but do not hide new occurrences."""
    source = tmp_path / "module.py"
    source.write_text('def first():\n    value: int = "bad"\n')
    original = diagnostic(ratchet, tmp_path)
    source.write_text('\n\ndef first():\n    value: int = "bad"\n')
    shifted = diagnostic(ratchet, tmp_path, line=4)
    added, removed = ratchet.compare([original], [shifted])
    assert not added and not removed
    added, _ = ratchet.compare([original], [shifted, shifted])
    assert sum(added.values()) == 1
    source.write_text('def second():\n    value: int = "bad"\n')
    changed_scope = diagnostic(ratchet, tmp_path)
    added, removed = ratchet.compare([original], [changed_scope])
    assert sum(added.values()) == sum(removed.values()) == 1
    added, removed = ratchet.compare([original], [])
    assert not added and sum(removed.values()) == 1


@pytest.mark.parametrize("tool", ["pyright", "pylint"])
def test_other_analyzer_reports_keep_rule_message_and_location(
    ratchet: ModuleType, tmp_path: Path, tool: str
) -> None:
    """All supported machine-readable formats retain enough diagnostic identity."""
    (tmp_path / "module.py").write_text('def first():\n    value: int = "bad"\n')
    record: dict[str, object] = {"message": "bad type"}
    if tool == "pyright":
        record.update(
            file=str(tmp_path / "module.py"),
            range={"start": {"line": 1}},
            rule="reportAssignmentType",
            severity="error",
        )
        raw = json.dumps({"generalDiagnostics": [record]})
    else:
        record.update(path="module.py", line=2, type="error")
        record["message-id"] = "E0001"
        raw = json.dumps([record])
    parsed = ratchet.parse_diagnostics(tool, raw, tmp_path)
    assert parsed[0]["file"] == "module.py"
    assert parsed[0]["scope"] == "first"
    assert parsed[0]["message"] == "bad type"
    assert parsed[0]["rule"]


def commit_file(root: Path, path: str) -> None:
    """Create local baseline history with no remote writes."""
    subprocess.run(["git", "-C", str(root), "add", path], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )


def test_project_duplicate_code_ignores_incidental_reporting_module(
    ratchet: ModuleType,
) -> None:
    """Duplicate-code identity follows its real file pair and text, not visit order."""
    row = {
        "tool": "pylint",
        "rule": "R0801",
        "file": "last_module.py",
        "scope": "<module>",
        "source": "unrelated header",
        "message": "Similar lines in 2 files ==first:[lines] ==second:[lines] value = 1",
    }
    relocated = {**row, "file": "other_module.py", "source": "other header"}
    added, removed = ratchet.compare([row], [relocated])
    assert not added and not removed
    added, _ = ratchet.compare([row], [relocated, relocated])
    assert sum(added.values()) == 1
    changed = {**relocated, "message": "Similar lines ==first ==third value = 1"}
    added, removed = ratchet.compare([row], [changed])
    assert sum(added.values()) == sum(removed.values()) == 1


def test_committed_baseline_cannot_grow_or_restore_removed_debt(
    ratchet: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing the manifest cannot approve extra occurrences or undo a reduction."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "module.py").write_text('def first():\n    value: int = "bad"\n')
    row = diagnostic(ratchet, tmp_path)
    baseline = tmp_path / "baseline.json"
    original = {"schema": 1, "diagnostics": [row]}
    baseline.write_text(json.dumps(original))
    commit_file(tmp_path, "baseline.json")
    ratchet.check_baseline_growth(tmp_path, baseline, original)
    with pytest.raises(ratchet.RatchetError, match="Baseline growth"):
        ratchet.check_baseline_growth(
            tmp_path, baseline, {"schema": 1, "diagnostics": [row, row]}
        )
    baseline.write_text(json.dumps({"schema": 1, "diagnostics": []}))
    commit_file(tmp_path, "baseline.json")
    monkeypatch.setenv("STATIC_ANALYSIS_BASE_REF", "HEAD")
    with pytest.raises(ratchet.RatchetError, match="Baseline growth"):
        ratchet.check_baseline_growth(tmp_path, baseline, original)


@pytest.mark.parametrize("returncode,output", [(127, ""), (1, ""), (1, "not json")])
def test_analyzer_failure_never_counts_as_zero_debt(
    ratchet: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    output: str,
) -> None:
    """Missing tools, crashes and malformed reports fail closed."""

    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, output, "tool failure")

    monkeypatch.setattr(ratchet.subprocess, "run", failed)
    with pytest.raises((ratchet.RatchetError, ValueError)):
        ratchet.analyze(tmp_path, "src", tmp_path / "reports", mypy_strict=True)


def test_ci_requires_committed_baseline_history(
    ratchet: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shallow or uncommitted baseline cannot silently bypass the growth guard."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setenv("CI", "true")
    with pytest.raises(ratchet.RatchetError, match="history is missing"):
        ratchet.check_baseline_growth(
            tmp_path, tmp_path / "baseline.json", {"schema": 1, "diagnostics": []}
        )
