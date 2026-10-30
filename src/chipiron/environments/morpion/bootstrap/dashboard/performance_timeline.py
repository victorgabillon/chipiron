"""Interval arithmetic for the existing report; spans are not causal critical paths."""

import math
from collections import defaultdict
from typing import Any


def number(value: object) -> float | None:
    """Keep absent, malformed and nonfinite measurements unknown."""
    return (
        float(value)
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def interval(row: dict[str, Any]) -> tuple[float, float] | None:
    """Only consistent, complete wall-clock evidence participates in interval arithmetic."""
    start, end = number(row.get("started_unix_s")), number(row.get("finished_unix_s"))
    elapsed = number(row.get("elapsed_s"))
    if (
        start is None
        or end is None
        or end < start
        or (elapsed is not None and elapsed < 0)
    ):
        return None
    if elapsed is not None and abs(end - start - elapsed) > max(0.05, elapsed * 0.001):
        return None
    return start, end


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Count overlapping and nested wall intervals once."""
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def interval_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed bounds, coverage and simultaneous distinct worker processes."""
    intervals = [i for row in records if (i := interval(row)) is not None]
    start = min((i[0] for i in intervals), default=None)
    end = max((i[1] for i in intervals), default=None)
    span = end - start if start is not None and end is not None else None
    covered = union_seconds(intervals) if intervals else None
    events: list[tuple[float, int, str]] = []
    for row in records:
        observed = interval(row)
        pid = row.get("process_identity") or row.get("pid")
        if (
            observed is None
            or observed[1] <= observed[0]
            or pid is None
            or row.get("stage") in {"wait", "worker_lifetime", "startup", "ownership"}
        ):
            continue
        events.extend([(observed[0], 1, str(pid)), (observed[1], -1, str(pid))])
    active: dict[str, int] = defaultdict(int)
    previous: float | None = None
    overlap = 0.0
    max_workers = 0
    for instant, delta, pid in sorted(events):
        if previous is not None and len(active) >= 2:
            overlap += instant - previous
        active[pid] += delta
        if active[pid] == 0:
            del active[pid]
        max_workers = max(max_workers, len(active))
        previous = instant
    detailed = [
        i
        for row in records
        if row.get("stage") != "worker_lifetime" and (i := interval(row)) is not None
    ]
    return {
        "unattributed_to_stage_s": span - union_seconds(detailed)
        if span is not None
        else None,
        "observed_start_unix_s": start,
        "observed_end_unix_s": end,
        "observed_span_s": span,
        "covered_interval_s": covered,
        "unattributed_within_span_s": span - covered
        if span is not None and covered is not None
        else None,
        "observed_worker_overlap_s": overlap if events else None,
        "max_observed_workers": max_workers if events else None,
        "complete_intervals": len(intervals),
        "observations": len(records),
        "missing_or_inconsistent_intervals": len(records) - len(intervals),
        "scope": "Recorded endpoints only; incomplete attempts and unrecorded startup/waits may extend beyond these bounds. Not a causal critical path.",
    }


def timeline_rows(
    records: list[dict[str, Any]], warnings: list[str]
) -> list[dict[str, Any]]:
    """Expose identities, clock quality and parent residuals without fabricating causes."""
    rows = []
    for record in records:
        observed = interval(record)
        start, end = (
            number(record.get("started_unix_s")),
            number(record.get("finished_unix_s")),
        )
        elapsed = number(record.get("elapsed_s"))
        clock = "complete" if observed is not None else "missing interval"
        if start is not None and end is not None and observed is None:
            clock = "inconsistent clocks"
            warnings.append(
                f"{record['measurement_id']}: inconsistent wall/elapsed clock evidence"
            )
        elif observed is not None and elapsed is None:
            clock = "wall interval only; monotonic elapsed unavailable"
        children = [
            child
            for child in records
            if child.get("parent_id") == record["measurement_id"]
        ]
        child_intervals = [
            i for child in children if (i := interval(child)) is not None
        ]
        residual = None
        if observed is not None and children:
            clipped = [
                (max(observed[0], a), min(observed[1], b))
                for a, b in child_intervals
                if b >= observed[0] and a <= observed[1]
            ]
            residual = observed[1] - observed[0] - union_seconds(clipped)
            if any(
                a < observed[0] - 0.05 or b > observed[1] + 0.05
                for a, b in child_intervals
            ):
                warnings.append(
                    f"{record['measurement_id']}: child interval extends outside parent"
                )
        rows.append({
            "Generation": record["generation"],
            "Stage": record["stage"],
            "Start (Unix s)": start,
            "End (Unix s)": end,
            "Elapsed (s)": elapsed if elapsed is not None and elapsed >= 0 else None,
            "Clock evidence": clock,
            "Evidence status": record.get("evidence_status"),
            "PID": record.get("pid"),
            "Process identity": record.get("process_identity"),
            "Attempt": record.get("attempt_id"),
            "Claim": record.get("claim_id"),
            "Source Dataset generation": record.get("source_dataset_generation"),
            "Source tree generation": record.get("source_tree_generation")
            if record.get("source_tree_generation") is not None
            else record.get("source_generation", record.get("input_tree_generation")),
            "Training generation": record.get("training_generation"),
            "Model generation": record.get("model_generation"),
            "Publication": record.get("publication_id"),
            "Dataset attempt": record.get("dataset_attempt_id"),
            "Source publication": record.get(
                "source_publication_id", record.get("source_tree_publication_id")
            ),
            "Evaluator": record.get("evaluator_name"),
            "Patch ID": record.get("patch_id"),
            "Measurement": record["measurement_id"],
            "Parent": record.get("parent_id"),
            "Unattributed within parent (s)": residual,
            "Process CPU (s)": number(record.get("cpu_s")),
            "Reaped child CPU (s)": number(record.get("children_cpu_s")),
            "RSS process lifetime peak (MiB)": number(
                record.get("process_peak_rss_mb")
            ),
            "Device timing scope": record.get(
                "device_timing_scope", "legacy/unknown; host wall is not device elapsed"
            ),
            "Wait reason": record.get("reason"),
            "Sources": record["sources"],
            "Ownership wait (s)": number(record.get("wait_s")),
            "Ownership held (s)": number(record.get("held_s")),
            "Observed incomplete wait (s)": number(record.get("observed_elapsed_s"))
            if record.get("evidence_status") == "incomplete"
            else None,
            "Observed through (Unix s)": number(record.get("last_observed_unix_s")),
            "Timing scope": record.get("timing_scope"),
            "Clock source": record.get("clock_source"),
            "Retry evidence": record.get("retry_evidence"),
            "Outcome": record.get("outcome"),
        })
    return rows
