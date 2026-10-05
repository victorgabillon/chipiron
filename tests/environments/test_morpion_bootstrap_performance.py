"""Observational metrics, compatible reads, and tiny scientific parity checks."""

from __future__ import annotations

import json
import random
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import performance
from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
    load_performance_summary,
)
from chipiron.environments.morpion.bootstrap.performance import (
    StageMeasurement,
    persist_stage_measurement,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineEvaluatorTrainingResult,
    pipeline_evaluator_training_result_from_dict,
    pipeline_evaluator_training_result_to_dict,
)
from tests.environments.test_morpion_bootstrap_pipeline_stages import (
    FakeMorpionSearchRunner,
    _artifact_pipeline_args,
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_boundary_metrics_preserve_rng_and_roundtrip(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """CPU metrics need no CUDA, preserve RNG, and survive actual JSON storage."""
    monkeypatch.setattr(performance, "current_rss_mb", lambda: 123.5)
    monkeypatch.setattr(performance, "available_ram_mb", lambda: 456.0)
    monkeypatch.setattr(performance, "process_peak_rss_mb", lambda: 200.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    observation = StageMeasurement(cuda_device="auto").finish(node_count=42)
    assert random.getstate() == random_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert observation["elapsed_s"] >= 0
    assert observation["finished_unix_s"] >= observation["started_unix_s"]
    assert observation["rss_before_mb"] == observation["rss_after_mb"] == 123.5
    assert observation["process_peak_rss_mb"] == 200
    assert observation["available_ram_after_mb"] == 456
    assert observation["cuda_before"] is observation["cuda_after"] is None
    persist_stage_measurement(tmp_path, 39, "checkpoint", observation)
    [path] = (tmp_path / "pipeline/performance/generation_000039").glob("*.json")
    saved = json.loads(path.read_text())
    assert saved.items() >= observation.items()
    assert saved["schema_version"] == 1
    assert saved["generation"] == 39
    assert saved["stage"] == "checkpoint"
    result = MorpionPipelineEvaluatorTrainingResult(
        final_loss=1, elapsed_s=2, model_bundle_path="model", performance=observation
    )
    assert (
        pipeline_evaluator_training_result_from_dict(
            json.loads(json.dumps(pipeline_evaluator_training_result_to_dict(result)))
        )
        == result
    )
    assert (
        pipeline_evaluator_training_result_from_dict({
            "final_loss": 1,
            "elapsed_s": 2,
            "model_bundle_path": "old",
        }).performance
        == {}
    )


def test_cuda_boundaries_without_a_gpu(monkeypatch: MonkeyPatch) -> None:
    """Only evaluator boundaries synchronize/reset; allocator units stay bytes."""
    calls: list[str] = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda device: calls.append("sync"),
        reset_peak_memory_stats=lambda device: calls.append("reset"),
        get_device_name=lambda device: "test GPU",
        get_device_properties=lambda device: SimpleNamespace(total_memory=1000),
        memory_allocated=lambda device: 10,
        memory_reserved=lambda device: 20,
        max_memory_allocated=lambda device: 30,
        max_memory_reserved=lambda device: 40,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, device=str))
    observation = StageMeasurement(cuda_device="auto").finish()
    assert calls == ["sync", "reset", "sync"]
    assert observation["cuda_before"]["allocated_bytes"] == 10
    assert observation["cuda_after"]["max_reserved_bytes"] == 40
    calls.clear()
    StageMeasurement(cuda_device="cpu").finish()
    assert not calls


def test_metrics_io_failure_does_not_fail_science(tmp_path: Path, caplog: Any) -> None:
    """Unwritable metrics must not fail or retry completed scientific work."""
    (tmp_path / "pipeline").write_text("not a directory")
    persist_stage_measurement(tmp_path, 1, "growth", {})
    assert "Could not persist" in caplog.text


def _tiny_pipeline(work_dir: Path) -> tuple[dict[str, Any], list[int]]:
    """Exercise actual dataset extraction and training on a deterministic exported row."""
    torch.manual_seed(174)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    args = _artifact_pipeline_args(work_dir)
    run_pipeline_growth_stage(args, runner, max_cycles=1)
    run_pipeline_dataset_stage(args, generation=1)
    manifest = run_pipeline_training_stage(args, generation=1)
    status = json.loads(
        (work_dir / "pipeline/generation_000001/training_status.json").read_text()
    )
    return {
        "selected": manifest.selected_evaluator_name,
        "results": status["evaluator_results"],
    }, runner.grow_calls


def test_tiny_pipeline_metrics_and_scientific_parity(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Measured and unmeasured executions yield identical data, losses and selection."""
    observed, observed_calls = _tiny_pipeline(tmp_path / "observed")
    summary = load_performance_summary(tmp_path / "observed")
    [row] = summary.generations
    for stage in ("growth", "checkpoint", "export", "dataset", "training", "cycle"):
        assert row[f"{stage} (s)"] >= 0
    assert row["Nodes"] == 5
    assert row["Dataset rows"] == 1
    assert row["RSS process peak (MiB)"] > 0
    assert row["GPU allocated peak (MiB)"] is None
    assert len(summary.evaluators) == len(observed["results"])
    assert all(
        result["performance"]["elapsed_s"] >= 0
        for result in observed["results"].values()
    )
    assert sum(bool(result["Selected?"]) for result in summary.evaluators) == 1
    monkeypatch.setattr(StageMeasurement, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(StageMeasurement, "finish", lambda self, **kw: {})
    from chipiron.environments.morpion.bootstrap import cycle_training
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    monkeypatch.setattr(stages, "persist_stage_measurement", lambda *a, **kw: None)
    monkeypatch.setattr(
        cycle_training, "persist_stage_measurement", lambda *a, **kw: None
    )
    plain, plain_calls = _tiny_pipeline(tmp_path / "plain")
    assert observed_calls == plain_calls
    assert observed["selected"] == plain["selected"]
    for evaluator, result in observed["results"].items():
        assert {
            k: v for k, v in result.items() if k not in {"elapsed_s", "performance"}
        } == {
            k: v
            for k, v in plain["results"][evaluator].items()
            if k not in {"elapsed_s", "performance"}
        }
    assert (tmp_path / "plain/rows/generation_000001.jsonl").read_bytes() == (
        tmp_path / "observed/rows/generation_000001.jsonl"
    ).read_bytes()


def test_summary_partial_legacy_and_overlap(tmp_path: Path) -> None:
    """No fabricated peaks for old metrics, and overlapping stages are not summed as wall time."""
    directory = tmp_path / "pipeline/generation_000038"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({
            "metadata": {
                "tree": {"node_count": 242680, "growth_elapsed_s": 590.923},
                "checkpoint": {"total_s": 93.759},
            }
        })
    )
    [old] = load_performance_summary(tmp_path).generations
    assert old["growth (s)"] == 590.923
    assert old["checkpoint (s)"] == 93.759
    assert old["RSS process peak (MiB)"] is None
    assert old["training (s)"] is None
    persist_stage_measurement(
        tmp_path,
        39,
        "growth",
        {"started_unix_s": 1, "finished_unix_s": 11, "elapsed_s": 10},
    )
    persist_stage_measurement(
        tmp_path,
        39,
        "dataset",
        {"started_unix_s": 6, "finished_unix_s": 16, "elapsed_s": 10},
    )
    summary = load_performance_summary(tmp_path)
    assert summary.generations[-1]["Timeline span (s)"] == 15
    assert len(summary.timeline) == 2
    (directory / "training_status.json").write_text("{")
    assert load_performance_summary(tmp_path).warnings


def test_performance_operations_is_lazy_and_partial_safe(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The actual Operations page renders partial metrics without heavy readers."""
    from chipiron.environments.morpion.bootstrap.dashboard import history_view
    from tests.environments.test_morpion_bootstrap_operator_ui import _app

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Performance requested full scientific artifacts")

    monkeypatch.setattr(
        history_view, "build_morpion_bootstrap_dashboard_data", forbidden
    )
    persist_stage_measurement(
        tmp_path, 39, "growth", {"elapsed_s": 2, "node_count_after": 42}
    )
    historical = tmp_path / "pipeline/generation_000038"
    historical.mkdir(parents=True)
    (historical / "manifest.json").write_text(
        json.dumps({
            "metadata": {
                "tree": {"node_count": 242680, "growth_elapsed_s": 590.923},
                "checkpoint": {"total_s": 93.759},
                "memory": {"rss_mb": 8486.19},
            }
        })
    )
    app = _app(tmp_path).run()
    assert not app.exception
    app.radio(key="bootstrap_operator_page").set_value("Operations").run()
    next(
        w for w in app.checkbox if w.label == "Load performance measurements"
    ).check().run()
    assert not app.exception
    assert any("growth (s)" in frame.value.columns for frame in app.dataframe)
