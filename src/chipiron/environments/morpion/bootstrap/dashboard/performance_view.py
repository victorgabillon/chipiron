"""Small-artifact performance summaries, independent of search and model loading."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .performance_evidence import EvidenceReader
from .performance_timeline import interval_summary, timeline_rows
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
    "activation",
)
MAX_GENERATIONS = 256


@dataclass(frozen=True)
class PerformanceSummary:
    """Tabular observations; None means unavailable, never an inferred zero."""

    generations: list[dict[str, Any]]
    evaluators: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    warnings: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)


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
    reader = EvidenceReader(work_dir, warnings)
    reader.load_training()
    reader.load_runtime()
    work_dir = reader.root
    pipeline = work_dir / "pipeline"
    directories = {p.name: p for p in pipeline.glob("generation_*") if p.is_dir()}
    perf_directories = {
        p.name: p for p in (pipeline / "performance").glob("generation_*") if p.is_dir()
    }
    names = sorted(
        set(directories)
        | set(perf_directories)
        | {f"generation_{g:06d}" for g in reader.records if g is not None}
    )
    for name in names[-MAX_GENERATIONS:]:
        try:
            generation = int(name.removeprefix("generation_"))
        except ValueError:
            continue
        reader.performance_files(work_dir, generation)
        if generation not in reader.training:
            status = reader.read(pipeline / name / "training_status.json")
            reader.training[generation] = status
            evidence_status = (
                "legacy completed; acceptance unavailable"
                if status.get("status") == "done"
                else str(status.get("status", "incomplete"))
            )
            if mapping(reader.authority.get("metadata")).get("durable_training"):
                evidence_status = "unaccepted projection"
            if reader.publications.get(generation, {}).get("accepted_training"):
                evidence_status = "unverified projection"
            reader.training_status(
                generation,
                status,
                f"pipeline/{name}/training_status.json",
                {"training_generation": generation, "evidence_status": evidence_status},
            )
    reader.load_activation()
    names = sorted(
        set(names) | {f"generation_{g:06d}" for g in reader.records if g is not None}
    )
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
        manifest = reader.read(directory / "manifest.json")
        dataset = reader.read(directory / "dataset_status.json")
        training = reader.training.get(generation, {})
        metadata = mapping(manifest.get("metadata"))
        tree = mapping(metadata.get("tree"))
        if metadata.get("round_performance") and not any(
            record["stage"] == "training" for record in reader.records[generation]
        ):
            reader.training_status(
                generation,
                {"metadata": metadata, "status": manifest.get("training_status")},
                f"pipeline/{name}/manifest.json",
                {
                    "training_generation": generation,
                    "evidence_status": "unverified projection"
                    if mapping(reader.authority.get("metadata")).get("durable_training")
                    else "legacy projection; acceptance unavailable",
                },
            )
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
            "Training core (s)": _number(
                mapping(training.get("metadata")).get("training_duration_s")
            ),
            "Accepted Training (s)": None,
        }
        row.update({
            "growth (s)": tree.get("growth_elapsed_s"),
            "cycle (s)": tree.get("cycle_elapsed_s"),
            "checkpoint (s)": mapping(metadata.get("checkpoint")).get("total_s"),
            "export (s)": mapping(metadata.get("training_export")).get("total_s"),
        })
        records = reader.records[generation]
        records.sort(key=lambda r: _number(r.get("started_unix_s")) or 0)
        stage_totals: dict[str, float] = {}
        evaluator_records: dict[str, dict[str, Any]] = {}
        rss: list[object] = []
        rss_after: list[object] = []
        available: list[float] = []
        gpu: list[object] = []
        for record in records:
            stage = record.get("stage")
            elapsed = _number(record.get("elapsed_s"))
            if (
                stage in STAGES
                and elapsed is not None
                and elapsed >= 0
                and not record.get("parent_id")
            ):
                stage_totals[stage] = stage_totals.get(stage, 0) + elapsed
            if stage == "growth":
                row.update({
                    "Nodes": record.get("node_count_after"),
                    "+Nodes": record.get("nodes_added"),
                    "Nodes/s": record.get("nodes_added_per_second"),
                })
            if stage == "training":
                row["Dataset rows"] = record.get("dataset_rows", row["Dataset rows"])
            if (
                stage == "evaluator"
                and isinstance(record.get("evaluator_name"), str)
                and (
                    record.get("evidence_status") == "accepted"
                    or record["evaluator_name"] not in evaluator_records
                )
            ):
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
        row.update({f"{stage} (s)": elapsed for stage, elapsed in stage_totals.items()})
        accepted_rounds = [
            elapsed
            for record in records
            if record["stage"] == "training"
            and record.get("evidence_status") == "accepted"
            and (elapsed := _number(record.get("elapsed_s"))) is not None
            and elapsed >= 0
        ]
        row["Accepted Training (s)"] = sum(accepted_rounds) if accepted_rounds else None
        row["RSS process peak (MiB)"] = _maximum(rss)
        row["RSS after (MiB)"] = _maximum(rss_after) or row["RSS after (MiB)"]
        row["Available RAM min (MiB)"] = min(available) if available else None
        gpu_peak = _maximum(gpu)
        row["GPU allocated peak (MiB)"] = (
            gpu_peak / 1024**2 if gpu_peak is not None else None
        )
        diagnostics = interval_summary(records)
        row["Timeline span (s)"] = diagnostics["observed_span_s"]
        row["Covered interval (s)"] = diagnostics["covered_interval_s"]
        row["Unattributed within span (s)"] = diagnostics["unattributed_within_span_s"]
        row["Observed worker overlap (s)"] = diagnostics["observed_worker_overlap_s"]
        row["Missing/inconsistent intervals"] = diagnostics[
            "missing_or_inconsistent_intervals"
        ]
        row["Training evidence"] = (
            "; ".join(
                sorted({
                    r["evidence_status"] for r in records if r["stage"] == "training"
                })
            )
            or "unavailable"
        )
        timeline.extend(timeline_rows(records, warnings))
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
                "Attempt": observation.get("attempt_id"),
                "Evidence status": observation.get("evidence_status"),
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
    timeline.extend(timeline_rows(reader.records[None], warnings))
    timeline.sort(
        key=lambda row: (
            row["Start (Unix s)"] is None,
            row["Start (Unix s)"] or 0,
            row["Measurement"],
        )
    )
    all_records = [
        record for row in generations for record in reader.records[row["Generation"]]
    ] + reader.records[None]
    diagnostics = interval_summary(all_records)
    diagnostics["incomplete_attempts"] = sum(
        a["evidence_status"] == "incomplete" for a in reader.attempts
    )
    diagnostics["retained_intervals_complete"] = (
        not warnings
        and not diagnostics["missing_or_inconsistent_intervals"]
        and not diagnostics["incomplete_attempts"]
        and bool(all_records)
    )
    diagnostics["evidence_complete"] = diagnostics[
        "retained_intervals_complete"
    ] and all(reader.records[row["Generation"]] for row in generations)
    diagnostics["limits"] = [
        "Only retained bounded evidence is included; absence is not zero or proof of failure.",
        "Stage totals are measured work and may nest/overlap; never sum them as run wall time.",
        "Overlap uses distinct observed process identities/PIDs and consistent Unix clocks; legacy PID reuse and cross-host clock alignment are unverified. Enclosing coordinator intervals may overlap child work; this is not CPU utilization.",
        "Unattributed time describes missing interval coverage, not a diagnosed wait cause.",
        "Process CPU excludes children; reaped child CPU is separate and may omit live children. RSS peaks are process-lifetime peaks, not stage peaks.",
    ]
    return PerformanceSummary(
        generations, evaluators, timeline, warnings, diagnostics, reader.attempts
    )


def render_performance(st: Any, work_dir: Path) -> None:
    """Render small persisted metrics with explicit overlap and peak semantics."""
    import pandas as pd

    summary = load_performance_summary(work_dir)
    for warning in summary.warnings:
        st.warning(warning)
    st.caption(
        "Seconds and MiB. Missing measurements are N/A. RSS is a process lifetime high water mark; CUDA peaks cover each evaluator/inference window and exclude other processes. Cycle is Growth's cycle only. Stages overlap: their sum is not end-to-end time. Timeline span covers recorded endpoints only; unrecorded waits and incomplete attempts can extend it. It is not a causal critical path."
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
    st.write(summary.diagnostics)
    if summary.attempts:
        with st.expander("Training attempts · acceptance and completeness"):
            st.dataframe(summary.attempts, hide_index=True, width="stretch")
    with st.expander("Stage timeline · worker overlap and patch provenance"):
        st.dataframe(summary.timeline, hide_index=True, width="stretch")
