"""Human-readable training logs for Morpion bootstrap pipelines."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

LOGGER = logging.getLogger(__name__)
TRAINING_LOG_SEPARATOR = "━" * 70


@dataclass(frozen=True, slots=True)
class TrainingActiveModelCursorSummary:
    """Human-facing summary of active-model and training-cursor provenance."""

    active_model_generation: int | None
    active_model_source_generation: int | None
    active_model_source: str
    cursor_started_generation: int | None
    cursor_completed_generation: int | None
    local_lower_bound_generation: int


def format_generation_list(generations: Sequence[int]) -> str:
    """Render a compact generation list for human logs."""
    if not generations:
        return "none"
    return ",".join(str(generation) for generation in generations)


def format_seconds(seconds: float) -> str:
    """Render elapsed seconds with stable precision."""
    return f"{seconds:.1f}s"


def format_percent(value: float | None) -> str:
    """Render one percentage with stable precision."""
    if value is None or not math.isfinite(value):
        return "unknown"
    return f"{value:.1f}"


def format_rows_per_second(rows_seen: int, elapsed_s: float) -> str:
    """Render throughput for progress logs."""
    if elapsed_s <= 0.0:
        return "unknown"
    return f"{rows_seen / elapsed_s:.1f}"


def format_training_evaluator_names(evaluator_names: Sequence[str]) -> str:
    """Render evaluator names as a comma-separated list."""
    if not evaluator_names:
        return "none"
    return ",".join(evaluator_names)


def format_training_dataset_summary(
    *,
    rows_path: str | Path,
    row_count: int | None,
    row_format: str,
    chunk_size: int | None,
    max_rows: int | None,
) -> str:
    """Render the high-level dataset summary fields."""
    effective_rows = _effective_row_count(row_count=row_count, max_rows=max_rows)
    chunk_count = _chunk_count(row_count=effective_rows, chunk_size=chunk_size)
    return (
        f"dataset={rows_path} rows={_format_optional_int(effective_rows)} "
        f"format={row_format} chunk_size={_format_chunk_size(chunk_size)} "
        f"chunks={_format_optional_int(chunk_count)} "
        f"max_rows={_format_max_rows(max_rows)}"
    )


def log_training_cycle_start(
    *,
    generation: int,
    rows_path: str | Path,
    row_count: int | None,
    row_format: str,
    chunk_size: int | None,
    evaluator_names: Sequence[str],
    active_model_cursor_summary: TrainingActiveModelCursorSummary,
    max_rows: int | None = None,
) -> None:
    """Log a high-level summary before training starts."""
    LOGGER.info(TRAINING_LOG_SEPARATOR)
    LOGGER.info("[training-cycle] generation=%s status=STARTED", generation)
    LOGGER.info(
        "[training-cycle] %s",
        format_training_dataset_summary(
            rows_path=rows_path,
            row_count=row_count,
            row_format=row_format,
            chunk_size=chunk_size,
            max_rows=max_rows,
        ),
    )
    LOGGER.info(
        "[training-cycle] evaluators=%s names=%s",
        len(evaluator_names),
        format_training_evaluator_names(evaluator_names),
    )
    LOGGER.info(
        "[training-cycle] active_model source=%s source_generation=%s "
        "active_generation=%s local_lower_bound=%s cursor_started=%s "
        "cursor_completed=%s",
        active_model_cursor_summary.active_model_source,
        _format_optional_int(active_model_cursor_summary.active_model_source_generation),
        _format_optional_int(active_model_cursor_summary.active_model_generation),
        active_model_cursor_summary.local_lower_bound_generation,
        _format_optional_int(active_model_cursor_summary.cursor_started_generation),
        _format_optional_int(active_model_cursor_summary.cursor_completed_generation),
    )
    LOGGER.info(TRAINING_LOG_SEPARATOR)


def log_training_evaluator_start(
    *,
    generation: int,
    evaluator_name: str,
    evaluator_index: int,
    evaluator_count: int,
    row_count: int | None,
    chunk_count: int | None,
) -> None:
    """Log a high-level evaluator start line."""
    LOGGER.info(
        "[training-evaluator] generation=%s evaluator=%s index=%s/%s "
        "status=STARTED rows=%s chunks=%s",
        generation,
        evaluator_name,
        evaluator_index,
        evaluator_count,
        _format_optional_int(row_count),
        _format_optional_int(chunk_count),
    )


def log_training_progress(
    *,
    generation: int,
    evaluator_name: str,
    chunk_index: int,
    chunk_count: int | None,
    rows_seen: int,
    row_count: int | None,
    elapsed_s: float,
    epoch_index: int | None = None,
    epoch_count: int | None = None,
) -> None:
    """Log throttled streaming progress through rows and chunks."""
    percent = None
    if row_count is not None and row_count > 0:
        percent = min(100.0, rows_seen * 100.0 / row_count)
    epoch_fields = ""
    if epoch_index is not None and epoch_count is not None:
        epoch_fields = f" epoch={epoch_index}/{epoch_count}"
    LOGGER.info(
        "[training-progress] generation=%s evaluator=%s%s chunk=%s/%s "
        "rows_seen=%s/%s percent=%s elapsed=%s rows_per_s=%s",
        generation,
        evaluator_name,
        epoch_fields,
        chunk_index,
        _format_optional_int(chunk_count),
        rows_seen,
        _format_optional_int(row_count),
        format_percent(percent),
        format_seconds(elapsed_s),
        format_rows_per_second(rows_seen, elapsed_s),
    )


def log_training_evaluator_done(
    *,
    generation: int,
    evaluator_name: str,
    evaluator_index: int,
    evaluator_count: int,
    elapsed_s: float,
    row_count: int | None,
    output: str | Path,
    metrics: Mapping[str, object] | None = None,
) -> None:
    """Log a high-level evaluator completion line."""
    LOGGER.info(
        "[training-evaluator] generation=%s evaluator=%s index=%s/%s "
        "status=DONE elapsed=%s rows=%s output=%s%s",
        generation,
        evaluator_name,
        evaluator_index,
        evaluator_count,
        format_seconds(elapsed_s),
        _format_optional_int(row_count),
        output,
        _format_metrics_suffix(metrics),
    )


def log_training_evaluator_failed(
    *,
    generation: int,
    evaluator_name: str,
    evaluator_index: int,
    evaluator_count: int,
    elapsed_s: float,
    error: BaseException,
) -> None:
    """Log a high-level evaluator failure line."""
    LOGGER.info(
        "[training-evaluator] generation=%s evaluator=%s index=%s/%s "
        "status=FAILED elapsed=%s error_type=%s error=%s",
        generation,
        evaluator_name,
        evaluator_index,
        evaluator_count,
        format_seconds(elapsed_s),
        type(error).__name__,
        error,
    )


def log_training_cycle_done(
    *,
    generation: int,
    elapsed_s: float,
    evaluators_done: int,
    evaluator_count: int,
    row_count: int | None,
    outputs: str | Path,
    active_model_candidate: str | Path | None,
) -> None:
    """Log a high-level training-cycle completion summary."""
    LOGGER.info(TRAINING_LOG_SEPARATOR)
    LOGGER.info(
        "[training-cycle] generation=%s status=DONE elapsed=%s "
        "evaluators_done=%s/%s rows=%s",
        generation,
        format_seconds(elapsed_s),
        evaluators_done,
        evaluator_count,
        _format_optional_int(row_count),
    )
    LOGGER.info(
        "[training-cycle] outputs=%s active_model_candidate=%s",
        outputs,
        "none" if active_model_candidate is None else active_model_candidate,
    )
    LOGGER.info(TRAINING_LOG_SEPARATOR)


def log_training_cycle_idle(
    *,
    reason: str,
    pending_generations: Sequence[int],
    claimable_generations: Sequence[int],
    local_lower_bound: int,
    active_model_source_generation: int | None,
) -> None:
    """Log a human-readable idle summary for training workers."""
    LOGGER.info(
        "[training-cycle] status=IDLE reason=%s pending_generations=%s "
        "claimable_generations=%s local_lower_bound=%s "
        "active_model_source_generation=%s",
        reason,
        format_generation_list(pending_generations),
        format_generation_list(claimable_generations),
        local_lower_bound,
        _format_optional_int(active_model_source_generation),
    )


def training_chunk_count(*, row_count: int | None, chunk_size: int | None) -> int | None:
    """Return the number of chunks a row source will produce."""
    return _chunk_count(row_count=row_count, chunk_size=chunk_size)


def should_log_training_progress(
    *,
    chunk_index: int,
    chunk_count: int | None,
) -> bool:
    """Return whether one chunk should emit a progress heartbeat."""
    if chunk_count is None:
        return chunk_index == 1 or chunk_index % 5 == 0
    if chunk_count <= 10:
        return True
    return chunk_index == 1 or chunk_index % 5 == 0 or chunk_index == chunk_count


def _effective_row_count(*, row_count: int | None, max_rows: int | None) -> int | None:
    if row_count is None:
        return max_rows
    if max_rows is None:
        return row_count
    return min(row_count, max_rows)


def _chunk_count(*, row_count: int | None, chunk_size: int | None) -> int | None:
    if row_count is None:
        return None
    if chunk_size is None:
        return 1
    if row_count == 0:
        return 0
    return math.ceil(row_count / chunk_size)


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "none"
    return str(value)


def _format_chunk_size(chunk_size: int | None) -> str:
    if chunk_size is None:
        return "all"
    return str(chunk_size)


def _format_max_rows(max_rows: int | None) -> str:
    if max_rows is None:
        return "all"
    return str(max_rows)


def _format_metrics_suffix(metrics: Mapping[str, object] | None) -> str:
    if metrics is None:
        return ""
    parts: list[str] = []
    for metric_key, log_key in (
        ("final_loss", "loss_final"),
        ("loss_mean", "loss_mean"),
        ("train_batches", "train_batches"),
        ("train_loss", "train_loss"),
        ("validation_loss", "validation_loss"),
    ):
        value = metrics.get(metric_key)
        if value is not None:
            parts.append(f"{log_key}={value}")
    if not parts:
        return ""
    return " " + " ".join(parts)
