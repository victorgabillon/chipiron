"""Small-artifact performance summaries, independent of search and model loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .view_model import mapping

if TYPE_CHECKING:
    from pathlib import Path

STAGES = (
    "growth",
    "checkpoint",
    "export",
    "dataset",
    "training",
    "reevaluation",
    "patch_apply",
    "cycle",
)
MAX_GENERATIONS = 256
MAX_OBSERVATIONS = 2048
MAX_ARTIFACT_BYTES = 256 * 1024


@dataclass(frozen=True)
class PerformanceSummary:
    """Tabular observations; None means unavailable, never an inferred zero."""

    generations: list[dict[str, Any]]
    evaluators: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    warnings: list[str]


def _read(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Read bounded JSON only, tolerating old, missing or incomplete artifacts."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_ARTIFACT_BYTES + 1)
        if len(raw) > MAX_ARTIFACT_BYTES:
            warnings.append(f"Oversize metrics artifact skipped: {path.name}")
            return {}
        return mapping(json.loads(raw))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError) as exc:
        warnings.append(f"{path.name}: {exc}")
        return {}


def _number(value: object) -> float | None:
    """Reject bool/nonfinite/malformed optional numeric fields."""
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _maximum(values: list[object]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return max(numbers) if numbers else None


def load_performance_summary(work_dir: Path) -> PerformanceSummary:
    """Join independent worker observations by generation, without loading a tree."""
    warnings: list[str] = []
    pipeline = work_dir / "pipeline"
    directories = {p.name: p for p in pipeline.glob("generation_*") if p.is_dir()}
    perf_directories = {
        p.name: p for p in (pipeline / "performance").glob("generation_*") if p.is_dir()
    }
    names = sorted(set(directories) | set(perf_directories))
    if len(names) > MAX_GENERATIONS:
        warnings.append(f"Showing the latest {MAX_GENERATIONS} generations.")
    generations: list[dict[str, Any]] = []
    evaluators: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for name in names[-MAX_GENERATIONS:]:
        try:
            generation = int(name.removeprefix("generation_"))
        except ValueError:
            continue
        directory = directories.get(name, pipeline / name)
        manifest = _read(directory / "manifest.json", warnings)
        dataset = _read(directory / "dataset_status.json", warnings)
        training = _read(directory / "training_status.json", warnings)
        metadata = mapping(manifest.get("metadata"))
        tree = mapping(metadata.get("tree"))
        row: dict[str, Any] = {
            "Generation": generation,
            "Nodes": tree.get("node_count"),
            "+Nodes": tree.get("nodes_added"),
            "Nodes/s": None,
            "Dataset rows": mapping(dataset.get("metadata")).get("dataset_rows"),
            **{f"{stage} (s)": None for stage in STAGES},
            "RSS after (MiB)": mapping(metadata.get("memory")).get("rss_mb"),
            "Available RAM min (MiB)": None,
            "RSS process peak (MiB)": None,
            "GPU allocated peak (MiB)": None,
            "Timeline span (s)": None,
        }
        row.update({
            "growth (s)": tree.get("growth_elapsed_s"),
            "cycle (s)": tree.get("cycle_elapsed_s"),
            "checkpoint (s)": mapping(metadata.get("checkpoint")).get("total_s"),
            "export (s)": mapping(metadata.get("training_export")).get("total_s"),
        })
        records = []
        files = (
            sorted(perf_directories[name].glob("*.json"))
            if name in perf_directories
            else []
        )
        if len(files) > MAX_OBSERVATIONS:
            warnings.append(
                f"Generation {generation}: observations truncated to {MAX_OBSERVATIONS}; totals are partial."
            )
        for path in files[-MAX_OBSERVATIONS:]:
            record = _read(path, warnings)
            if (
                record.get("schema_version") == 1
                and record.get("generation") == generation
            ):
                records.append(record)
        records.sort(key=lambda r: _number(r.get("started_unix_s")) or 0)
        stage_totals: dict[str, float] = {}
        evaluator_records: dict[str, dict[str, Any]] = {}
        rss: list[object] = []
        rss_after: list[object] = []
        available: list[float] = []
        gpu: list[object] = []
        starts: list[float] = []
        ends: list[float] = []
        for record in records:
            stage = record.get("stage")
            elapsed = _number(record.get("elapsed_s"))
            if stage in STAGES and elapsed is not None:
                stage_totals[stage] = stage_totals.get(stage, 0) + elapsed
            if stage == "growth":
                row.update({
                    "Nodes": record.get("node_count_after"),
                    "+Nodes": record.get("nodes_added"),
                    "Nodes/s": record.get("nodes_added_per_second"),
                })
            if stage == "training":
                row["Dataset rows"] = record.get("dataset_rows")
            if stage == "evaluator" and isinstance(record.get("evaluator_name"), str):
                evaluator_records[record["evaluator_name"]] = record
            rss.append(record.get("process_peak_rss_mb"))
            rss_after.append(record.get("rss_after_mb"))
            for key in ("available_ram_before_mb", "available_ram_after_mb"):
                value = _number(record.get(key))
                if value is not None:
                    available.append(value)
            cuda = mapping(record.get("cuda_after")) or mapping(
                mapping(record.get("inference_window")).get("cuda_after")
            )
            gpu.append(cuda.get("max_allocated_bytes"))
            start, end = (
                _number(record.get("started_unix_s")),
                _number(record.get("finished_unix_s")),
            )
            if start is not None and end is not None:
                starts.append(start)
                ends.append(end)
            timeline.append({
                "Generation": generation,
                "Stage": stage,
                "Start (Unix s)": start,
                "End (Unix s)": end,
                "Elapsed (s)": elapsed,
                "PID": record.get("pid"),
                "Evaluator": record.get("evaluator_name"),
                "Model generation": record.get("model_generation"),
                "Patch ID": record.get("patch_id"),
            })
        row.update({f"{stage} (s)": elapsed for stage, elapsed in stage_totals.items()})
        row["RSS process peak (MiB)"] = _maximum(rss)
        row["RSS after (MiB)"] = _maximum(rss_after) or row["RSS after (MiB)"]
        row["Available RAM min (MiB)"] = min(available) if available else None
        gpu_peak = _maximum(gpu)
        row["GPU allocated peak (MiB)"] = (
            gpu_peak / 1024**2 if gpu_peak is not None else None
        )
        row["Timeline span (s)"] = max(ends) - min(starts) if starts else None
        results = mapping(training.get("evaluator_results"))
        for evaluator in sorted(set(results) | set(evaluator_records)):
            result = mapping(results.get(evaluator))
            observation = evaluator_records.get(
                evaluator, mapping(result.get("performance"))
            )
            metrics = mapping(observation.get("metrics")) or result
            cuda = mapping(observation.get("cuda_after"))
            allocated = _number(cuda.get("max_allocated_bytes"))
            evaluators.append({
                "Generation": generation,
                "Evaluator": evaluator,
                "Training (s)": observation.get("elapsed_s", result.get("elapsed_s")),
                "RSS process peak (MiB)": observation.get("process_peak_rss_mb"),
                "GPU allocated peak (MiB)": allocated / 1024**2
                if allocated is not None
                else None,
                "Train loss": metrics.get("train_loss"),
                "Validation loss": metrics.get("validation_loss"),
                "Train MAE": metrics.get("train_mae"),
                "Validation MAE": metrics.get("validation_mae"),
                "Selected?": evaluator == training.get("selected_evaluator_name")
                if training.get("selected_evaluator_name")
                else None,
            })
        generations.append(row)
    return PerformanceSummary(generations, evaluators, timeline, warnings)


def render_performance(st: Any, work_dir: Path) -> None:
    """Render small persisted metrics with explicit overlap and peak semantics."""
    import pandas as pd

    summary = load_performance_summary(work_dir)
    for warning in summary.warnings:
        st.warning(warning)
    st.caption(
        "Seconds and MiB. Missing measurements are N/A. RSS is a process lifetime high water mark; CUDA peaks cover each evaluator/inference window and exclude other processes. Cycle is Growth's cycle only. Stages overlap: their sum is not end-to-end time. Timeline span includes waiting and retries, not a computed causal critical path."
    )
    if not summary.generations:
        st.info("No persisted performance evidence yet.")
        return
    st.dataframe(
        pd.DataFrame(summary.generations).style.format(na_rep="N/A", precision=3),
        hide_index=True,
        width="stretch",
    )
    chart_data = pd.DataFrame(summary.generations).apply(pd.to_numeric, errors="coerce")
    axis = st.selectbox("Scaling axis", ["Generation", "Nodes", "Dataset rows"])
    st.line_chart(
        chart_data,
        x=axis,
        y=[f"{stage} (s)" for stage in STAGES if stage != "cycle"],
    )
    left, right = st.columns(2)
    with left:
        st.line_chart(
            chart_data,
            x="Nodes",
            y=["RSS process peak (MiB)", "GPU allocated peak (MiB)"],
        )
    with right:
        st.line_chart(chart_data, x="Nodes", y="Nodes/s")
    st.subheader("Per-evaluator training")
    if summary.evaluators:
        st.dataframe(
            pd.DataFrame(summary.evaluators).style.format(na_rep="N/A", precision=3),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Per-evaluator measurements are unavailable for this run.")
    with st.expander("Stage timeline · worker overlap and patch provenance"):
        st.dataframe(summary.timeline, hide_index=True, width="stretch")
