"""Fail-closed partition, artifact, routing and same-tree PR reuse contracts."""

import copy
import importlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
# tests/scripts is a real package. Load the CI namespace privately so neither
# collection nor these tests change application/pytest import-path precedence.
package = ModuleType("ci_speed_fixture")
package.__path__ = [str(ROOT / "scripts")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    "ci_speed_fixture.ci_validation", ROOT / "scripts/ci_validation.py"
)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
coverage_gate = importlib.import_module("ci_speed_fixture.ci_coverage")
PARTITION, SHARD_COUNT, shard_for = gate.PARTITION, gate.SHARD_COUNT, gate.shard_for
pytestmark = pytest.mark.fast


def execution(ids: list[str], shard: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make complete receipt fixtures; real plugin execution is exercised separately."""
    worker = {"selected": ids, "deselected": [], "shard": shard}
    return {
        "selected_ids": ids,
        "deselected_ids": [],
        "collect_only": False,
        "exit_code": 0,
        "execution_errors": [],
        "shard": shard,
        "reports": [
            {"nodeid": n, "phase": p, "outcome": "passed", "duration": 0.1}
            for n in ids
            for p in ("setup", "call", "teardown")
        ],
        "worker_collections": {
            "gw0": copy.deepcopy(worker),
            "gw1": copy.deepcopy(worker),
        },
    }


def full_ids() -> list[str]:
    """Use two real file names in every partition, including parameterized cases."""
    buckets: dict[int, list[str]] = {i: [] for i in range(SHARD_COUNT)}
    i = 0
    while any(len(names) < 2 for names in buckets.values()):
        name = f"test_partition_{i}.py"
        bucket = buckets[shard_for(name, SHARD_COUNT)]
        if len(bucket) < 2:
            bucket.append(name)
        i += 1
    return sorted(
        f"{name}::test_ok[{p}]"
        for names in buckets.values()
        for name in names
        for p in range(2)
    )


@pytest.fixture
def evidence() -> dict[str, Any]:
    """Build a fully bound PR/main pair with distinct SHAs and an equal tree."""
    source = {
        "candidate": "a" * 40,
        "tree": "b" * 40,
        "parents": ["c" * 40, "d" * 40],
        "git_objects": {p: "e" * 40 for p in gate.INPUTS},
    }
    contract = gate.read_json(ROOT / ".github/ci-source-contracts.json")
    packages = contract["packages"] | contract.get("candidate_package_overrides", {})
    identity = {
        name: {
            "members": len(p["files"]),
            "origin": f"/env/site-packages/{p['module']}/__init__.py",
        }
        for name, p in packages.items()
    }
    identity["chipiron"] = {
        "reference": source["candidate"],
        "mode": "build-candidate",
        "editable": False,
        "source_members": 539,
        "package_members": 577,
    }
    ids = full_ids()
    q: dict[str, Any] = {
        "schema": 1,
        "source": source,
        "run_id": "7",
        "run_attempt": "1",
        "event": "pull_request",
        "complete": True,
        "lanes": {},
        "coverage": {"sha256": "f" * 64},
        "reference": {
            "collect_only": True,
            "exit_code": 0,
            "reports": [],
            "selected_ids": ids,
            "deselected_ids": [],
            "shard": None,
        },
    }
    artifacts = []
    for lane in gate.LANES:
        shard = None
        selected = [f"{lane}::test_aux"]
        if lane in gate.FULL_LANES:
            index = gate.FULL_LANES.index(lane)
            shard = {
                "index": index,
                "count": SHARD_COUNT,
                "algorithm": PARTITION,
                "full_selected_ids": ids,
                "full_deselected_ids": [],
            }
            selected = [n for n in ids if shard_for(n, SHARD_COUNT) == index]
        metrics = execution(selected, shard)
        if shard is not None:
            deselected = [n for n in ids if n not in selected]
            metrics["deselected_ids"] = deselected
            for worker in metrics["worker_collections"].values():
                worker["deselected"] = deselected
        artifact = {
            "id": len(artifacts) + 1,
            "name": f"artifact-{lane}",
            "digest": "sha256:" + "1" * 64,
        }
        artifacts.append({**artifact, "expired": False})
        files = {
            name: "2" * 64
            for name in (
                "candidate.json",
                "identity-before.json",
                "identity-after.json",
                "dependencies.json",
                "full-tests.json" if shard else f"{lane}-tests.json",
            )
        }
        if shard:
            files.update({
                name: "3" * 64
                for name in (".coverage", "full-coverage.json", "full.xml")
            })
        if lane == "full-0":
            files.update({"chipiron.whl": "4" * 64, "chipiron.tar.gz": "5" * 64})
        q["lanes"][lane] = {
            "candidate": {
                **source,
                "lane": lane,
                "run_id": "7",
                "run_attempt": "1",
                "event": "pull_request",
            },
            "identity-before": copy.deepcopy(identity),
            "identity-after": copy.deepcopy(identity),
            "dependencies": [
                {"name": n, "version": p["version"]} for n, p in packages.items()
            ],
            "metrics": metrics,
            "files": files,
            "artifact": artifact,
        }
    now = datetime(2026, 11, 16, 12, tzinfo=UTC)
    return {
        "main": {**source, "candidate": "6" * 40},
        "pr": {
            "number": 17,
            "merged": True,
            "merge_commit_sha": "6" * 40,
            "head": {"sha": "d" * 40, "repo": {"id": 19}},
            "base": {"ref": "main", "repo": {"id": 19}},
        },
        "run": {
            "id": 7,
            "run_attempt": 1,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "workflow_id": 23,
            "path": ".github/workflows/ci.yaml",
            "repository": {"id": 19},
            "head_repository": {"id": 19},
            "head_sha": "d" * 40,
            "pull_requests": [{"number": 17}],
            "updated_at": now.isoformat(),
        },
        "jobs": [
            {"name": name, "run_id": 7, "status": "completed", "conclusion": "success"}
            for name in sorted(gate.REQUIRED_JOBS)
        ],
        "q": q,
        "tested": {
            "sha": source["candidate"],
            "tree": {"sha": source["tree"]},
            "parents": [{"sha": p} for p in source["parents"]],
        },
        "artifacts": artifacts,
        "head_prs": [{"number": 17}],
        "repository_id": 19,
        "workflow_id": 23,
        "now": now,
    }


def test_distinct_commit_same_reviewed_merge_tree_reuses(
    evidence: dict[str, Any],
) -> None:
    """Source equivalence, not commit equality, permits a normal reviewed merge."""
    assert evidence["main"]["candidate"] != evidence["q"]["source"]["candidate"]
    gate.check_reuse(**evidence)
    evidence["run"]["pull_requests"] = []
    gate.check_reuse(**evidence)
    for association in ([], [{"number": 18}], [{"number": 17}, {"number": 18}]):
        evidence["head_prs"] = association
        with pytest.raises(ValueError, match="tested-head PR association"):
            gate.check_reuse(**evidence)


@pytest.mark.parametrize(
    "fault",
    [
        "base_advanced",
        "squash",
        "rebase",
        "tree",
        "configuration",
        "dependency",
        "identity_missing",
        "identity_changed",
        "artifact_missing",
        "artifact_expired",
        "artifact_digest",
        "failed_job",
        "skipped_job",
        "missing_job",
        "cancelled_run",
        "partial",
        "wrong_attempt",
        "fork",
        "foreign_workflow",
        "ambiguous_pr",
        "expired",
        "failed_phase",
        "missing_teardown",
        "duplicate_execution",
        "wrong_selection",
        "wrong_reference",
        "worker_disagreement",
        "missing_lane",
        "coverage",
        "distribution",
    ],
)
def test_reuse_refuses_incomplete_or_changed_evidence(
    evidence: dict[str, Any], fault: str
) -> None:
    """Every uncertain scientific or GitHub identity forces new FULL validation."""
    q = evidence["q"]
    lane = q["lanes"]["full-0"]
    if fault == "base_advanced":
        evidence["main"]["parents"][0] = "9" * 40
    elif fault in {"squash", "rebase"}:
        evidence["main"]["parents"] = ["c" * 40]
    elif fault == "tree":
        evidence["main"]["tree"] = "9" * 40
    elif fault == "configuration":
        evidence["main"]["git_objects"] = {
            **evidence["main"]["git_objects"],
            "pyproject.toml": "9" * 40,
        }
    elif fault == "dependency":
        lane["dependencies"][0]["version"] = "0.0.0"
    elif fault == "identity_missing":
        del lane["identity-before"]["algorhino-anemone"]
    elif fault == "identity_changed":
        lane["identity-after"]["chipiron"]["reference"] = "9" * 40
    elif fault == "artifact_missing":
        evidence["artifacts"].pop()
    elif fault == "artifact_expired":
        evidence["artifacts"][0]["expired"] = True
    elif fault == "artifact_digest":
        evidence["artifacts"][0]["digest"] = "sha256:" + "9" * 64
    elif fault == "failed_job":
        evidence["jobs"][0]["conclusion"] = "failure"
    elif fault == "skipped_job":
        evidence["jobs"][0]["conclusion"] = "skipped"
    elif fault == "missing_job":
        evidence["jobs"].pop()
    elif fault == "cancelled_run":
        evidence["run"]["conclusion"] = "cancelled"
    elif fault == "partial":
        q["complete"] = False
    elif fault == "wrong_attempt":
        q["run_attempt"] = "2"
    elif fault == "fork":
        evidence["pr"]["head"]["repo"]["id"] = 20
    elif fault == "foreign_workflow":
        evidence["run"]["workflow_id"] = 24
    elif fault == "ambiguous_pr":
        evidence["run"]["pull_requests"].append({"number": 18})
    elif fault == "expired":
        evidence["now"] += timedelta(days=8)
    elif fault == "failed_phase":
        lane["metrics"]["reports"][0]["outcome"] = "failed"
    elif fault == "missing_teardown":
        lane["metrics"]["reports"].pop()
    elif fault == "duplicate_execution":
        lane["metrics"]["reports"].append(lane["metrics"]["reports"][0])
    elif fault == "wrong_selection":
        lane["metrics"]["selected_ids"].pop()
    elif fault == "wrong_reference":
        q["reference"]["selected_ids"] = q["reference"]["selected_ids"][1:]
    elif fault == "worker_disagreement":
        lane["metrics"]["worker_collections"]["gw1"]["selected"] = []
    elif fault == "missing_lane":
        del q["lanes"]["pr"]
    elif fault == "coverage":
        del q["coverage"]
    else:
        del lane["files"]["chipiron.whl"]
    with pytest.raises((KeyError, ValueError)):
        gate.check_reuse(**evidence)


def test_main_advance_requires_new_equivalent_candidate(
    evidence: dict[str, Any],
) -> None:
    """A new base can reuse only new evidence with that exact merge topology."""
    evidence["main"]["parents"] = ["9" * 40, "d" * 40]
    with pytest.raises(ValueError):
        gate.check_reuse(**evidence)
    evidence["q"] = copy.deepcopy(evidence["q"])
    evidence["q"]["source"]["parents"] = evidence["main"]["parents"]
    for lane in evidence["q"]["lanes"].values():
        lane["candidate"]["parents"] = evidence["main"]["parents"]
    evidence["tested"]["parents"][0]["sha"] = "9" * 40
    gate.check_reuse(**evidence)


def test_artifact_seal_detects_missing_added_and_corrupt_files(tmp_path: Path) -> None:
    """Raw coverage and installed evidence cannot disappear or change in transit."""
    path = tmp_path / ".coverage"
    path.write_bytes(b"raw coverage")
    gate.seal(tmp_path)
    gate.check_seal(tmp_path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        gate.check_seal(tmp_path)
    path.unlink()
    with pytest.raises(ValueError):
        gate.check_seal(tmp_path)
    path.write_bytes(b"raw coverage")
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError):
        gate.check_seal(tmp_path)


def test_artifact_redirect_strips_github_credential() -> None:
    """The signed storage URL must not receive repository credentials."""
    req = urllib.request.Request(
        "https://api.github.com/archive", headers={"Authorization": "Bearer private"}
    )
    redirected = gate.ArtifactRedirect().redirect_request(
        req, None, 302, "Found", {}, "https://storage.example/archive"
    )
    assert redirected is not None and not redirected.has_header("Authorization")


def test_real_four_shard_xdist_selection_matches_unsharded(tmp_path: Path) -> None:
    """Resolve the actual workflow through Tox, then execute its real pytest argv."""
    for name in {n.split("::")[0] for n in full_ids()}:
        (tmp_path / name).write_text(
            "import pytest\n@pytest.mark.parametrize('x',[0,1])\ndef test_ok(x): assert x < 2\n@pytest.mark.skip(reason='setup skip')\ndef test_skip(): pass\n"
        )
    (tmp_path / "conftest.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def pytest_configure(config):\n"
        " if not hasattr(config, 'workerinput'):\n"
        "  Path('parsed.json').write_text(json.dumps({\n"
        "   'args': list(config.invocation_params.args),\n"
        "   'collect_only': config.option.collectonly,\n"
        "   'workers': config.option.numprocesses,\n"
        "   'dist': config.option.dist,\n"
        "   'marker': config.option.markexpr,\n"
        "   'index': config.getoption('ci_shard_index'),\n"
        "   'count': config.getoption('ci_shard_count')}))\n"
    )
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yaml").read_text(), Loader=yaml.BaseLoader
    )
    steps = workflow["jobs"]["pytest"]["steps"]
    tox = shutil.which("tox") or str(Path(sys.executable).with_name("tox"))

    def run(index: int | None) -> dict[str, Any]:
        name = (
            "Collect unchanged unsharded FULL reference"
            if index is None
            else "Test installed package"
        )
        shell = next(s["run"] for s in steps if s.get("name") == name)
        shell = shell.splitlines()[0].replace("${{ runner.temp }}", str(tmp_path))
        shell = shell.replace("${{ matrix.shard }}", str(index))
        invocation = shlex.split(shell)
        assert invocation[:3] == ["python", "-m", "tox"]
        configured = subprocess.run(
            [
                tox,
                "c",
                "--workdir",
                str(tmp_path / "tox"),
                "-k",
                "commands",
                *invocation[3:],
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        command = next(
            shlex.split(line.strip())
            for line in configured.stdout.splitlines()
            if line.strip().startswith("python scripts/ci_test_metrics.py ")
        )
        forwarded = invocation[invocation.index("--") + 1 :]
        assert command[-len(forwarded) :] == forwarded
        command[:2] = [sys.executable, str(ROOT / "scripts/ci_test_metrics.py")]
        result = subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        parsed = gate.read_json(tmp_path / "parsed.json")
        assert parsed == {
            "args": ["-p", "ci_test_metrics", *command[2:]],
            "collect_only": index is None,
            "workers": 0 if index is None else 2,
            "dist": "no" if index is None else "loadfile",
            "marker": "not integration and not external_data",
            "index": index,
            "count": None if index is None else SHARD_COUNT,
        }
        metrics = gate.read_json(tmp_path / ".ci-reports/tests.json")
        assert metrics["collect_only"] == (index is None)
        return metrics

    reference = run(None)
    assert reference["selected_ids"] == sorted([
        *full_ids(),
        *(f"{name}::test_skip" for name in {n.split("::")[0] for n in full_ids()}),
    ])
    seen: list[str] = []
    for index in range(SHARD_COUNT):
        metrics = run(index)
        gate.check_execution(metrics)
        assert metrics["shard"]["full_selected_ids"] == reference["selected_ids"]
        seen.extend(metrics["selected_ids"])
    assert Counter(seen) == Counter(reference["selected_ids"])
    assert all(count == 1 for count in Counter(seen).values())


@pytest.mark.parametrize(
    "event", ["schedule", "workflow_dispatch", "push", "pull_request"]
)
def test_route_uncertainty_never_skips_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    """Scheduled/manual/direct/ambiguous pushes and unknown PR changes stay complete."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(gate, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(gate, "source_identity", dict)
    monkeypatch.setattr(
        gate,
        "try_reuse",
        lambda source: (_ for _ in ()).throw(ValueError("No evidence")),
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "missing-event.json"))
    gate.route()
    assert (
        gate.read_json(tmp_path / ".ci-reports/validation-route.json")["mode"] == "full"
    )


def test_documentation_policy_checks_content_paths_modes_and_nonempty_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependencies, workflow, tests, renames, symlinks and unknown docs never qualify."""
    monkeypatch.chdir(tmp_path)

    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    run("init", "-q")
    run("config", "user.email", "ci@example.invalid")
    run("config", "user.name", "CI test")
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs/testing.md"
    doc.write_text("Original prose\n")
    run("add", ".")
    run("commit", "-qm", "base")
    base = run("rev-parse", "HEAD")
    assert not gate.documentation_only(base, base)
    doc.write_text("Updated prose\n")
    run("commit", "-qam", "docs")
    assert gate.documentation_only(base, "HEAD")
    doc.unlink()
    doc.symlink_to("elsewhere")
    run("add", ".")
    run("commit", "-qm", "symlink")
    assert not gate.documentation_only(base, "HEAD")
    for name in (
        "pyproject.toml",
        ".github/workflows/ci.yaml",
        "tests/test_unknown.py",
        "docs/unknown.md",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new input\n")
        run("add", ".")
        run("commit", "-qm", "stronger")
        assert not gate.documentation_only(base, "HEAD")


def test_workflow_preserves_resource_selection_identity_and_aggregate_contracts() -> (
    None
):
    """No marker/time/resource change is hidden in the job partition."""
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yaml").read_text(), Loader=yaml.BaseLoader
    )
    jobs = workflow["jobs"]
    job = jobs["pytest"]
    assert [x["lane"] for x in job["strategy"]["matrix"]["include"]] == list(gate.LANES)
    assert job["strategy"]["fail-fast"] == "false" and job["timeout-minutes"] == "120"
    steps = job["steps"]
    full = next(s for s in steps if s.get("name") == "Test installed package")
    assert full["timeout-minutes"] == "60"
    assert (
        "--ci-shard-count=4" in full["run"]
        and "-m " not in full["run"].split(" -- ", 1)[1]
    )
    verify = [s for s in steps if "verify-candidate" in s.get("run", "")]
    assert len(verify) == 2 and verify[-1]["if"] == "always()"
    assert "pip check" in verify[-1]["run"]
    assert jobs["test"]["if"] == "${{ always() }}"
    assert set(jobs["test"]["needs"]) == {
        "scope",
        "quality",
        "pytest",
        "static",
        "full-evidence",
    }
    assert "schedule" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow["on"]
    assert jobs["scope"]["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
    }
    assert jobs["full-evidence"]["permissions"] == {
        "contents": "read",
        "actions": "read",
    }
    for name, definition in jobs.items():
        if name not in {"scope", "full-evidence"}:
            assert "permissions" not in definition
        assert set(definition.get("permissions", {}).values()) <= {"read"}


@pytest.mark.parametrize("mode", ["full", "docs", "reuse"])
@pytest.mark.parametrize(
    "fault", [None, "scope", "quality", "pytest", "static", "full-evidence"]
)
def test_aggregate_shell_gate_rejects_interrupted_or_partial_results(
    tmp_path: Path, mode: str, fault: str | None
) -> None:
    """Execute the actual final shell gate, including skipped-job semantics."""
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yaml").read_text(), Loader=yaml.BaseLoader
    )
    command = workflow["jobs"]["test"]["steps"][0]["run"]
    statuses = {
        "scope": "success",
        "quality": "success",
        "pytest": "success",
        "static": "success",
        "full-evidence": "success",
    }
    if mode != "full":
        statuses.update(
            pytest="skipped", static="skipped", **{"full-evidence": "skipped"}
        )
    if fault is not None:
        statuses[fault] = "cancelled"
    names = {
        "scope": "SCOPE_RESULT",
        "quality": "QUALITY_RESULT",
        "pytest": "PYTEST_RESULT",
        "static": "STATIC_RESULT",
        "full-evidence": "EVIDENCE_RESULT",
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        env={
            **os.environ,
            "VALIDATION_MODE": mode,
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            **{names[k]: v for k, v in statuses.items()},
        },
        check=False,
    )
    assert (result.returncode == 0) == (fault is None)
    if fault is None and mode != "full":
        assert "No runtime tests executed" in (tmp_path / "summary").read_text()


def test_real_coverage_combine_preserves_lines_branches_and_child_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use actual SQLite coverage files, not a fabricated coverage success flag."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sample"
    source.mkdir()
    (source / "subject.py").write_text(
        "def choose(x):\n    if x:\n        return 1\n    return 0\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, subprocess\nfrom sample.subject import choose\n"
        "assert choose(int(sys.argv[1]) % 2) in (0, 1)\n"
        "subprocess.run([sys.executable, '-c', 'from sample.subject import choose; choose(1)'], check=True)\n"
    )
    rc = tmp_path / ".coveragerc"
    rc.write_text(
        "[run]\nbranch = true\nparallel = true\npatch = subprocess\nsource = sample\n"
    )
    inputs = tmp_path / "inputs"

    def coverage(*args: str, data: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", *args],
            env={**os.environ, "COVERAGE_FILE": str(data), "COVERAGE_RCFILE": str(rc)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    raw = []
    for index, lane in enumerate(gate.FULL_LANES):
        folder = inputs / lane
        folder.mkdir(parents=True)
        data = folder / ".coverage"
        coverage("run", str(driver), str(index), data=data)
        # One parent and one actual child-process report must be present.
        assert len(list(folder.glob(".coverage.*"))) == 2
        coverage("combine", data=data)
        coverage("json", "-o", str(folder / "full-coverage.json"), data=data)
        raw.append(str(data))
    data = tmp_path / ".coverage"
    combined = tmp_path / "combined.json"
    coverage("combine", "--keep", *raw, data=data)
    coverage("json", "-o", str(combined), data=data)
    parts = [inputs / lane for lane in gate.FULL_LANES]
    final = coverage_gate.verify_union(parts, data, combined, source, tmp_path / "src")
    assert any(f["executed_branches"] for f in final["files"].values())
    for fault in ("lines", "branches", "membership", "exclusions", "branch_mode"):
        broken = copy.deepcopy(final)
        file = next(iter(broken["files"].values()))
        if fault in {"lines", "branches"}:
            file[f"executed_{fault}"].pop()
        elif fault == "membership":
            broken["files"].popitem()
        elif fault == "exclusions":
            file["excluded_lines"].append(999)
        else:
            broken["meta"]["branch_coverage"] = False
        gate.write_json(combined, broken)
        with pytest.raises(ValueError):
            coverage_gate.verify_union(parts, data, combined, source, tmp_path / "src")


def test_assemble_requires_sealed_complete_same_run_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: dict[str, Any]
) -> None:
    """Exercise the actual seven-artifact assembly boundary and byte tamper refusal."""
    monkeypatch.chdir(tmp_path)
    contract = tmp_path / ".github/ci-source-contracts.json"
    contract.parent.mkdir()
    contract.write_bytes((ROOT / ".github/ci-source-contracts.json").read_bytes())
    q = evidence["q"]
    monkeypatch.setattr(gate, "source_identity", lambda: q["source"])
    monkeypatch.setenv("GITHUB_RUN_ID", q["run_id"])
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", q["run_attempt"])
    monkeypatch.setenv("GITHUB_EVENT_NAME", q["event"])
    inputs = tmp_path / "inputs"
    artifacts = []
    for lane, record in q["lanes"].items():
        name = f"ci-validation-{lane}-{q['source']['candidate']}-{q['run_id']}-{q['run_attempt']}"
        folder = inputs / name
        folder.mkdir(parents=True)
        for filename in record["files"]:
            (folder / filename).write_text("retained artifact")
        for key in ("candidate", "identity-before", "identity-after", "dependencies"):
            gate.write_json(folder / f"{key}.json", record[key])
        metric_name = "full" if lane in gate.FULL_LANES else lane
        gate.write_json(folder / f"{metric_name}-tests.json", record["metrics"])
        if lane == "full-0":
            gate.write_json(folder / "full-reference-tests.json", q["reference"])
        gate.seal(folder)
        artifacts.append({
            "name": name,
            "id": len(artifacts) + 1,
            "expired": False,
            "digest": "sha256:" + "f" * 64,
        })
    monkeypatch.setattr(gate, "artifacts_for", lambda run: artifacts)
    monkeypatch.setattr(
        gate,
        "download_json",
        lambda artifact, member: gate.read_json(inputs / artifact["name"] / member),
    )
    gate.assemble(inputs)
    assembled = gate.read_json(Path(".ci-reports/qualification.json"))
    assert assembled["complete"] is False
    gate.check_qualification(assembled, q["source"])
    monkeypatch.setattr(
        gate, "download_json", lambda artifact, member: {"changed": "0" * 64}
    )
    with pytest.raises(ValueError, match="authenticated manifest"):
        gate.assemble(inputs)
    (inputs / artifacts[0]["name"] / "identity-before.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.assemble(inputs)


def test_api_discovery_reuses_only_authenticated_successful_candidate(
    evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise bounded discovery, artifact naming and all downstream checks together."""
    q = evidence["q"]
    artifact = {
        "id": 99,
        "name": f"ci-qualification-{q['source']['candidate']}-7-1",
        "digest": "sha256:" + "3" * 64,
    }
    artifacts = [*evidence["artifacts"], artifact]
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "19")

    def api(path: str) -> Any:
        if path.startswith("commits/"):
            return [
                {
                    "number": 17,
                    "merge_commit_sha": evidence["main"]["candidate"],
                    "merged_at": "present",
                }
            ]
        if path == "pulls/17":
            return evidence["pr"]
        if path == "actions/workflows/ci.yaml":
            return {"id": 23}
        if path.startswith("actions/workflows/23/runs"):
            return {"total_count": 1, "workflow_runs": [evidence["run"]]}
        if path.startswith("actions/runs/7/artifacts"):
            return {"total_count": len(artifacts), "artifacts": artifacts}
        if path.startswith("git/commits/"):
            return evidence["tested"]
        if path.startswith("actions/runs/7/attempts/1/jobs"):
            return {"total_count": len(evidence["jobs"]), "jobs": evidence["jobs"]}
        raise AssertionError(path)

    monkeypatch.setattr(gate, "api", api)
    monkeypatch.setattr(gate, "download_json", lambda value: q)
    evidence["run"]["updated_at"] = datetime.now(UTC).isoformat()
    evidence["run"]["pull_requests"] = []  # Real post-merge GitHub response.
    result = gate.try_reuse(evidence["main"])
    assert (
        result["mode"] == "reuse"
        and result["tested_candidate"] == q["source"]["candidate"]
    )
    evidence["jobs"].pop()
    with pytest.raises(ValueError, match="No qualifying PR evidence"):
        gate.try_reuse(evidence["main"])
    artifacts.pop()
    with pytest.raises(ValueError, match="No unique qualification"):
        gate.try_reuse(evidence["main"])


def test_download_checks_github_archive_digest_before_reading_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository artifact name or JSON claim alone cannot authenticate bytes."""
    import io
    import zipfile
    from types import SimpleNamespace

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("qualification.json", '{"complete": true}')
    payload = data.getvalue()
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GH_TOKEN", "private-fixture")
    monkeypatch.setattr(
        gate.urllib.request,
        "build_opener",
        lambda handler: SimpleNamespace(
            open=lambda request, timeout: io.BytesIO(payload)
        ),
    )
    artifact = {
        "id": 7,
        "expired": False,
        "size_in_bytes": len(payload),
        "archive_download_url": "https://api.github.com/repos/owner/repo/actions/artifacts/7/zip",
        "digest": "sha256:" + gate.digest(payload),
    }
    assert gate.download_json(artifact) == {"complete": True}
    artifact["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        gate.download_json(artifact)


@pytest.mark.parametrize("fault", ["forbidden", "missing_digest", "expired_artifact"])
def test_unavailable_authenticated_reuse_routes_to_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Denied read permissions or unavailable artifact evidence never skip FULL."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(gate, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(gate, "source_identity", dict)

    def unavailable(source: dict[str, Any]) -> dict[str, Any]:
        if fault == "forbidden":
            raise urllib.error.HTTPError(
                "https://api.github.com/fixture", 403, "Forbidden", None, None
            )
        if fault == "missing_digest":
            raise KeyError("digest")
        message = "Expired scientific artifact"
        raise ValueError(message)

    monkeypatch.setattr(gate, "try_reuse", unavailable)
    gate.route()
    receipt = gate.read_json(tmp_path / ".ci-reports/validation-route.json")
    assert receipt["mode"] == "full"
    assert receipt["tests_executed_by_scope_job"] is False
    assert (tmp_path / "output").read_text() == "mode=full\n"


def test_parser_contract_restores_shared_capture_before_metrics_warning(
    tmp_path: Path,
) -> None:
    """Run the real producer/consumer order that failed in the hosted FULL-3 worker."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--no-cov",
            "-n",
            "0",
            "-q",
            "-o",
            f"cache_dir={tmp_path / 'cache'}",
            f"--basetemp={tmp_path / 'pytest'}",
            "tests/learning/test_parsley_consumer_compatibility.py::test_chess_launcher_extra_args_resolve_presets",
            "tests/environments/test_morpion_bootstrap_performance.py::test_metrics_io_failure_does_not_fail_science",
        ],
        cwd=ROOT,
        env={**os.environ, "CHIPIRON_OUTPUT_DIR": str(tmp_path / "output")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_coverage_membership_requires_raw_evidence_and_complete_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce the seven asymmetric installed/checkout paths with real raw DBs."""
    from coverage import Coverage, CoverageData

    monkeypatch.chdir(tmp_path)
    installed = tmp_path / "site/chipiron"
    checkout = tmp_path / "src/chipiron"
    installed.mkdir(parents=True)
    checkout.mkdir(parents=True)
    # Same seven path shapes and shard membership as retained run 44040060171.
    exclusive_installed = [
        "environments/chess/players/args/__init__.py",
        "environments/chess/players/args/chess_player_args.py",
        "environments/chess/players/wiring/__init__.py",
        "environments/chess/players/wiring/chess_wiring.py",
    ]
    exclusive_checkout = [
        "debug/tree_search_debug_selector.py",
        "environments/integer_reduction/players/evaluators/integer_reduction_state_evaluator.py",
        "environments/integer_reduction/players/evaluators/wiring.py",
    ]
    code = "def choose(value):\n    if value:\n        return 1\n    return 0\nignored = 0  # pragma: no cover\n"
    for name in [
        "common.py",
        "never_imported.py",
        *exclusive_installed,
        *exclusive_checkout,
    ]:
        for folder in (installed, checkout):
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("" if name.endswith("__init__.py") else code)
    (tmp_path / ".coveragerc").write_text("[run]\nbranch=true\n")
    parts = [tmp_path / f"full-{i}" for i in range(4)]
    raw = []
    for index, folder in enumerate(parts):
        folder.mkdir()
        cov = Coverage(data_file=str(folder / ".coverage"))
        cov.start()
        selected = [installed / "common.py"]
        if index == 0:
            selected += [installed / name for name in exclusive_installed]
            selected += [checkout / name for name in exclusive_checkout[1:]]
        elif index == 1:
            selected += [checkout / exclusive_checkout[0]]
        for path in selected:
            namespace: dict[str, Any] = {}
            exec(compile(path.read_bytes(), str(path), "exec"), namespace)
            if "choose" in namespace:
                namespace["choose"](1)
        cov.stop()
        # Limit the fixture to its actual sources, excluding this test's frame.
        data = CoverageData(basename=str(folder / ".coverage"))
        data.add_arcs({str(p): cov.get_data().arcs(str(p)) or [] for p in selected})
        data.write()
        cov = Coverage(data_file=str(folder / ".coverage"))
        cov.load()
        cov.json_report(
            morfs=[str(p) for p in selected], outfile=str(folder / "full-coverage.json")
        )
        gate.seal(folder)
        raw.append(str(folder / ".coverage"))
    cov = Coverage(data_file=str(tmp_path / ".coverage"))
    cov.combine(data_paths=raw, keep=True)
    cov.save()
    combined = tmp_path / "combined.json"
    cov.json_report(outfile=str(combined))

    def verify() -> dict[str, Any]:
        return dict(
            coverage_gate.verify_union(
                parts, tmp_path / ".coverage", combined, installed, checkout
            )
        )

    final = verify()
    sets = [set(gate.read_json(p / "full-coverage.json")["files"]) for p in parts]
    assert len(set.union(*sets) - set.intersection(*sets)) == 7
    never = final["files"]["site/chipiron/never_imported.py"]
    assert never["missing_lines"] and not never["executed_lines"]
    assert never["excluded_lines"]
    assert len(final["files"]) == len(list(installed.rglob("*.py"))) + 3
    # A reported execution cannot disappear merely because other shards have it.
    fragment = parts[0] / "full-coverage.json"
    saved = fragment.read_bytes()
    broken = gate.read_json(fragment)
    broken["files"].pop(
        next(name for name in broken["files"] if "chess_player_args" in name)
    )
    gate.write_json(fragment, broken)
    with pytest.raises(ValueError, match="raw/source"):
        verify()
    fragment.write_bytes(saved)
    # Extra/missing source, changed executable statements and altered exclusions.
    for mutation in ("extra", "missing", "lines", "exclusion", "total"):
        broken = gate.read_json(fragment)
        name = next(name for name in broken["files"] if "chess_wiring.py" in name)
        if mutation == "extra":
            broken["files"]["foreign.py"] = broken["files"][name]
        elif mutation == "missing":
            broken["files"].pop(name)
        elif mutation == "lines":
            broken["files"][name]["executed_lines"].append(999)
        elif mutation == "exclusion":
            broken["files"][name]["excluded_lines"] = []
        else:
            broken["totals"]["covered_lines"] += 1
        gate.write_json(fragment, broken)
        with pytest.raises(ValueError):
            verify()
        fragment.write_bytes(saved)
    # Raw mutation cannot be hidden behind unchanged percentages/reports.
    raw_path = parts[0] / ".coverage"
    saved_raw = raw_path.read_bytes()
    data = CoverageData(basename=str(raw_path))
    data.read()
    data.add_arcs({str(installed / "common.py"): [(2, 999)]})
    data.write()
    with pytest.raises(ValueError):
        verify()
    raw_path.write_bytes(saved_raw)
    # A self-consistent rewrite of raw data and JSON is still unauthenticated:
    # the aggregate must compare the seal with the GitHub-authenticated manifest.
    data = CoverageData(basename=str(raw_path))
    data.read()
    data.add_arcs({str(installed / "common.py"): [(2, 1)]})
    data.write()
    changed = Coverage(data_file=str(raw_path))
    changed.load()
    changed.json_report(outfile=str(fragment))
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.check_seal(parts[0])
    raw_path.write_bytes(saved_raw)
    fragment.write_bytes(saved)
    gate.check_seal(parts[0])
    raw_path.unlink()
    with pytest.raises(ValueError, match="Missing raw"):
        verify()
    raw_path.write_bytes(saved_raw)
    alias = checkout / exclusive_checkout[0]
    alias.write_text(code + "changed = True\n")
    with pytest.raises(ValueError, match="bytes changed"):
        verify()


@pytest.mark.parametrize("fault", ["source", "execution", "artifact"])
def test_coverage_gate_rechecks_qualification_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
    fault: str,
) -> None:
    """Coverage cannot turn a failed execution or changed receipt into qualification."""
    monkeypatch.chdir(tmp_path)
    q = copy.deepcopy(evidence["q"])
    q["complete"] = False
    contract = tmp_path / ".github/ci-source-contracts.json"
    contract.parent.mkdir()
    contract.write_bytes((ROOT / ".github/ci-source-contracts.json").read_bytes())
    current = copy.deepcopy(q["source"])
    if fault == "source":
        current["tree"] = "f" * 40
    elif fault == "execution":
        q["lanes"]["full-3"]["metrics"]["exit_code"] = 1
    monkeypatch.setattr(gate, "source_identity", lambda: current)
    monkeypatch.setattr(gate, "check_seal", lambda _root: {"altered": "0" * 64})
    gate.write_json(Path(".ci-reports/qualification.json"), q)
    with pytest.raises(ValueError):
        gate.check_coverage(tmp_path / "inputs", tmp_path / "combined.json")
    assert gate.read_json(Path(".ci-reports/qualification.json"))["complete"] is False
