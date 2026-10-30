"""Metadata-only regression for accepted Training, nested intervals and missing evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
    load_performance_summary,
)


def write(root: Path, name: str, value: dict) -> Path:
    """Write one canonical small fixture artifact."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def observation(
    root: Path,
    generation: int,
    stage: str,
    start: float,
    end: float,
    pid: int = 1,
    **extra: object,
) -> Path:
    """Persist a synthetic boundary measurement."""
    return write(
        root,
        f"pipeline/performance/generation_{generation:06d}/{stage}-{start}.json",
        {
            "schema_version": 1,
            "generation": generation,
            "stage": stage,
            "started_unix_s": start,
            "finished_unix_s": end,
            "elapsed_s": end - start,
            "pid": pid,
            **extra,
        },
    )


def fixture(root: Path) -> dict:
    """Materialize the accepted B2 layout without tensors or snapshots."""
    data = json.loads(
        (Path(__file__).parent / "fixtures/accepted_nested_training.json").read_text()
    )
    for name, value in data["files"].items():
        write(root, name, value)
    return data


def test_accepted_training_96_nested_layout_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover the missing round and deduplicate nested copies without writes."""
    fixture(tmp_path)
    before = {
        p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    original = Path.open

    def guarded(path: Path, mode: str = "r", *args: object, **kwargs: object) -> Any:
        assert mode in {"r", "rb"}, "report attempted mutation"
        assert (
            path.name
            in {
                "run_state.json",
                "manifest.json",
                "dataset_status.json",
                "training_status.json",
                "proposal.json",
                "admission.json",
            }
            or path.parent.parent.name == "performance"
        )
        assert path.suffix != ".pt" and "scientific_snapshot" not in path.parts
        return original(path, mode, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded)
        summary = load_performance_summary(tmp_path)
    [row] = summary.generations
    assert row["training (s)"] == pytest.approx(2387.7749312749947)
    assert row["Timeline span (s)"] == pytest.approx(2792.078563928604)
    assert row["Training evidence"] == "accepted"
    assert row["Missing/inconsistent intervals"] == 0
    evaluators = [r for r in summary.timeline if r["Stage"] == "evaluator"]
    assert len(evaluators) == len(summary.evaluators) == 9
    assert all(r["Parent"] and len(r["Sources"]) == 2 for r in evaluators)
    assert all(r["Source Dataset generation"] == 96 for r in evaluators)
    assert summary.diagnostics["observed_worker_overlap_s"] == 0
    assert not summary.warnings
    after = {
        p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_overlap_nested_and_unattributed_time(tmp_path: Path) -> None:
    """Compute interval unions and worker overlap without adding child durations."""
    observation(
        tmp_path,
        1,
        "training",
        10,
        30,
        pid=1,
        evaluator={
            "started_unix_s": 12,
            "finished_unix_s": 20,
            "elapsed_s": 8,
            "pid": 1,
        },
    )
    observation(tmp_path, 1, "growth", 15, 25, pid=2)
    observation(tmp_path, 1, "checkpoint", 35, 40, pid=2)
    summary = load_performance_summary(tmp_path)
    [row] = summary.generations
    assert row["training (s)"] == 20
    assert row["Timeline span (s)"] == 30
    assert row["Covered interval (s)"] == 25
    assert row["Unattributed within span (s)"] == 5
    assert row["Observed worker overlap (s)"] == 10
    parent = next(r for r in summary.timeline if r["Stage"] == "training")
    assert parent["Unattributed within parent (s)"] == 12


def test_legacy_missing_and_bad_clocks(tmp_path: Path) -> None:
    """Keep old missing durations unknown and reject inconsistent wall evidence."""
    write(
        tmp_path,
        "pipeline/generation_000001/manifest.json",
        {"metadata": {"tree": {"growth_elapsed_s": 10}}},
    )
    [legacy] = load_performance_summary(tmp_path).generations
    assert legacy["growth (s)"] == 10
    assert legacy["Timeline span (s)"] is None
    assert legacy["training (s)"] is None
    observation(tmp_path, 1, "growth", 20, 10)
    observation(tmp_path, 1, "dataset", 20, 30, elapsed_s=100)
    summary = load_performance_summary(tmp_path)
    assert summary.generations[0]["Timeline span (s)"] is None
    assert summary.diagnostics["missing_or_inconsistent_intervals"] == 2
    assert len(summary.warnings) == 2
    assert not summary.diagnostics["evidence_complete"]


def test_incomplete_and_failed_attempts_are_not_accepted(tmp_path: Path) -> None:
    """Keep attempt outcomes separate from accepted results and retry evidence."""
    fixture(tmp_path)
    write(
        tmp_path,
        f"training_publications_v1/staging/{'a' * 32}/admission.json",
        {"attempt_id": "a" * 32, "generation": 96},
    )
    write(
        tmp_path,
        f"training_publications_v1/staging/{'b' * 32}/proposal.json",
        {
            "admission": {"attempt_id": "b" * 32, "generation": 96},
            "status": {"generation": 96, "status": "failed"},
        },
    )
    summary = load_performance_summary(tmp_path)
    assert {a["evidence_status"] for a in summary.attempts} == {
        "accepted",
        "failed",
        "incomplete",
    }
    assert all(a["multiple_attempts_observed"] for a in summary.attempts)
    assert summary.generations[0]["training (s)"] == pytest.approx(2387.7749312749947)
    assert summary.diagnostics["incomplete_attempts"] == 1
    assert not summary.diagnostics["evidence_complete"]


def test_descriptor_mismatch_and_escaping_paths(tmp_path: Path) -> None:
    """Reject changed accepted evidence and paths outside the report workspace."""
    fixture(tmp_path)
    proposal = next(
        (tmp_path / "training_publications_v1/objects").glob("*/proposal.json")
    )
    proposal.write_text("{}")
    summary = load_performance_summary(tmp_path)
    assert not any(a["evidence_status"] == "accepted" for a in summary.attempts)
    assert any("digest mismatch" in w for w in summary.warnings)
    assert summary.generations[0]["Training evidence"] == "unverified projection"
    proposal.unlink()
    proposal.symlink_to("/etc/passwd")
    assert any(
        "escapes workspace" in w for w in load_performance_summary(tmp_path).warnings
    )


def test_activation_receipt_refresh_and_stage_total(tmp_path: Path) -> None:
    """Join refresh to its activation while retaining model/tree generation differences."""
    observation(tmp_path, 98, "activation", 20, 50)
    identifier = "c" * 32
    descriptor = {
        "model_activation": {
            "request_sha256": "request",
            "previous_head": "previous",
            "request": {
                "training_generation": 96,
                "candidate": {"generation": 96},
                "accepted_training": {"attempt_id": "a" * 32},
            },
            "refresh_performance": {
                "started_unix_s": 22,
                "finished_unix_s": 42,
                "elapsed_s": 20,
                "pid": 1,
            },
        }
    }
    path = write(
        tmp_path,
        f"generation_publications_v3/objects/{identifier}/.generation_publication_v3.json",
        descriptor,
    )
    write(
        tmp_path,
        "run_state.json",
        {
            "metadata": {
                "durable_generation_publication_v1": {
                    "generations": [
                        {
                            "generation": 98,
                            "id": identifier,
                            "activation_request_sha256": "request",
                            "descriptor_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                        }
                    ]
                }
            }
        },
    )
    summary = load_performance_summary(tmp_path)
    [row] = summary.generations
    assert row["activation (s)"] == 30
    assert row["Covered interval (s)"] == 30
    refresh = next(r for r in summary.timeline if r["Stage"] == "activation.refresh")
    assert refresh["Training generation"] == refresh["Model generation"] == 96
    assert refresh["Generation"] == 98
    assert refresh["Parent"]


def test_default_missing_round_and_nonfinite_artifacts(tmp_path: Path) -> None:
    """Completed legacy projections do not imply complete timing evidence."""
    write(
        tmp_path,
        "pipeline/generation_000001/training_status.json",
        {
            "status": "done",
            "metadata": {"training_duration_s": 12},
            "evaluator_results": {"legacy": {"elapsed_s": 12}},
        },
    )
    summary = load_performance_summary(tmp_path)
    assert summary.generations[0]["Training core (s)"] == 12
    assert summary.generations[0]["training (s)"] is None
    assert summary.generations[0]["Timeline span (s)"] is None
    assert not summary.diagnostics["evidence_complete"]
    assert any("round interval unavailable" in warning for warning in summary.warnings)
    path = tmp_path / "pipeline/generation_000001/manifest.json"
    path.write_text('{"metadata":{"tree":{"growth_elapsed_s":1e400}}}')
    summary = load_performance_summary(tmp_path)
    assert any("nonfinite" in warning for warning in summary.warnings)
    json.dumps(summary.generations, allow_nan=False)


def test_oversize_authority_and_bad_artifact_are_explicit(tmp_path: Path) -> None:
    """Bound metadata reads and report truncation instead of pretending acceptance."""
    (tmp_path / "run_state.json").write_bytes(b" " * (4 * 1024 * 1024 + 1))
    write(
        tmp_path, "pipeline/generation_000001/training_status.json", {"status": "done"}
    )
    summary = load_performance_summary(tmp_path)
    assert any("byte limit" in warning for warning in summary.warnings)
    assert not summary.diagnostics["evidence_complete"]


def test_manifest_only_legacy_round_is_not_labelled_accepted(tmp_path: Path) -> None:
    """Recover small legacy projections while keeping acceptance unknown."""
    write(
        tmp_path,
        "pipeline/generation_000001/manifest.json",
        {
            "training_status": "done",
            "metadata": {
                "round_performance": {
                    "started_unix_s": 10,
                    "finished_unix_s": 15,
                    "elapsed_s": 5,
                }
            },
        },
    )
    summary = load_performance_summary(tmp_path)
    assert summary.generations[0]["training (s)"] == 5
    assert summary.generations[0]["Accepted Training (s)"] is None
    assert (
        summary.timeline[0]["Evidence status"]
        == "legacy projection; acceptance unavailable"
    )


def test_legacy_explicit_source_generation_survives_missing_authority(
    tmp_path: Path,
) -> None:
    """An unavailable publication must not erase an observation's explicit source."""
    observation(tmp_path, 1, "dataset", 1, 2, input_tree_generation=0)
    [row] = load_performance_summary(tmp_path).timeline
    assert row["Source tree generation"] == 0
