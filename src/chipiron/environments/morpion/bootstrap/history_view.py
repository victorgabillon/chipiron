"""Dashboard-ready read-only history views for Morpion bootstrap runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .bootstrap_loop import MorpionBootstrapPaths
from .history import (
    MorpionBootstrapEvent,
    MorpionBootstrapLatestStatus,
    MorpionEvaluatorMetrics,
    load_bootstrap_history,
    load_latest_bootstrap_status,
)
from .record_status import current_record_score
from .run_state import MorpionBootstrapRunState, load_bootstrap_run_state


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRunView:
    """Loaded bootstrap artifacts for one Morpion work directory."""

    work_dir: Path
    run_state: MorpionBootstrapRunState | None
    latest_status: MorpionBootstrapLatestStatus
    history: tuple[MorpionBootstrapEvent, ...]


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRunSummary:
    """High-level summary of one Morpion bootstrap run."""

    num_cycles: int
    num_train_cycles: int
    num_no_train_cycles: int
    latest_cycle_index: int | None
    latest_generation: int | None
    latest_active_evaluator_name: str | None
    latest_tree_num_nodes: int | None
    latest_record_score: int | None
    latest_record_total_points: int | None
    latest_timestamp_utc: str | None


@dataclass(frozen=True, slots=True)
class IntTimeSeriesPoint:
    """One integer time-series point keyed by bootstrap cycle."""

    cycle_index: int
    generation: int
    timestamp_utc: str
    value: int


@dataclass(frozen=True, slots=True)
class OptionalIntTimeSeriesPoint:
    """One optional integer time-series point keyed by bootstrap cycle."""

    cycle_index: int
    generation: int
    timestamp_utc: str
    value: int | None


@dataclass(frozen=True, slots=True)
class OptionalFloatTimeSeriesPoint:
    """One optional float time-series point keyed by bootstrap cycle."""

    cycle_index: int
    generation: int
    timestamp_utc: str
    value: float | None


@dataclass(frozen=True, slots=True)
class ActiveEvaluatorTimeSeriesPoint:
    """One time-series point describing the active evaluator for a cycle."""

    cycle_index: int
    generation: int
    timestamp_utc: str
    active_evaluator_name: str | None


@dataclass(frozen=True, slots=True)
class TrainingTriggeredTimeSeriesPoint:
    """One time-series point describing whether training ran for a cycle."""

    cycle_index: int
    generation: int
    timestamp_utc: str
    triggered: bool


@dataclass(frozen=True, slots=True)
class EvaluatorSelectionSummary:
    """Summary of how the active evaluator changed over time."""

    latest_active_evaluator_name: str | None
    num_switches: int
    active_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class MorpionRecordProgressSummary:
    """Summary of canonical Morpion record progression across history."""

    latest_score: int | None
    best_score: int | None
    first_cycle_reaching_best: int | None
    latest_total_points: int | None
    best_total_points: int | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapDashboardData:
    """Bundled summaries and time series for future dashboard consumption."""

    run_summary: MorpionBootstrapRunSummary
    evaluator_selection_summary: EvaluatorSelectionSummary
    record_progress_summary: MorpionRecordProgressSummary
    tree_num_nodes: tuple[IntTimeSeriesPoint, ...]
    canonical_record_score: tuple[OptionalIntTimeSeriesPoint, ...]
    record_total_points: tuple[OptionalIntTimeSeriesPoint, ...]
    dataset_num_rows: tuple[OptionalIntTimeSeriesPoint, ...]
    evaluator_loss_by_name: Mapping[str, tuple[OptionalFloatTimeSeriesPoint, ...]]
    active_evaluator: tuple[ActiveEvaluatorTimeSeriesPoint, ...]


def load_morpion_bootstrap_run_view(
    work_dir: str | Path,
) -> MorpionBootstrapRunView:
    """Load the persisted bootstrap artifacts for one work directory."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    run_state = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.exists()
        else None
    )
    return MorpionBootstrapRunView(
        work_dir=paths.work_dir,
        run_state=run_state,
        latest_status=load_latest_bootstrap_status(paths.latest_status_path),
        history=load_bootstrap_history(paths.history_jsonl_path),
    )


def summarize_bootstrap_run(
    run_view: MorpionBootstrapRunView,
) -> MorpionBootstrapRunSummary:
    """Summarize the latest known bootstrap run state for dashboard display."""
    history = run_view.history
    latest_event = _latest_known_event(run_view)
    latest_run_state = run_view.run_state

    num_cycles = len(history)
    num_train_cycles = sum(1 for event in history if event.training.triggered)
    latest_record_status = (
        None if latest_event is None else latest_event.record
    )
    if latest_record_status is None and latest_run_state is not None:
        latest_record_status = latest_run_state.latest_record_status

    return MorpionBootstrapRunSummary(
        num_cycles=num_cycles,
        num_train_cycles=num_train_cycles,
        num_no_train_cycles=num_cycles - num_train_cycles,
        latest_cycle_index=_latest_cycle_index(run_view),
        latest_generation=_latest_generation(run_view),
        latest_active_evaluator_name=_latest_active_evaluator_name(run_view),
        latest_tree_num_nodes=_latest_tree_num_nodes(run_view),
        latest_record_score=None
        if latest_record_status is None
        else current_record_score(latest_record_status),
        latest_record_total_points=None
        if latest_record_status is None
        else latest_record_status.current_best_total_points,
        latest_timestamp_utc=None if latest_event is None else latest_event.timestamp_utc,
    )


def tree_num_nodes_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[IntTimeSeriesPoint, ...]:
    """Return tree-size time series points for every bootstrap cycle."""
    return tuple(
        IntTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            value=event.tree.num_nodes,
        )
        for event in history
    )


def canonical_record_score_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[OptionalIntTimeSeriesPoint, ...]:
    """Return canonical Morpion record-score points keyed by cycle."""
    return tuple(
        OptionalIntTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            value=current_record_score(event.record),
        )
        for event in history
    )


def record_total_points_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[OptionalIntTimeSeriesPoint, ...]:
    """Return total occupied points derived from recorded Morpion status."""
    return tuple(
        OptionalIntTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            value=event.record.current_best_total_points,
        )
        for event in history
    )


def dataset_num_rows_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[OptionalIntTimeSeriesPoint, ...]:
    """Return dataset-row counts by bootstrap cycle."""
    return tuple(
        OptionalIntTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            value=event.dataset.num_rows,
        )
        for event in history
    )


def training_triggered_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[TrainingTriggeredTimeSeriesPoint, ...]:
    """Return whether each cycle triggered export/training."""
    return tuple(
        TrainingTriggeredTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            triggered=event.training.triggered,
        )
        for event in history
    )


def evaluator_loss_series_by_name(
    history: Sequence[MorpionBootstrapEvent],
) -> Mapping[str, tuple[OptionalFloatTimeSeriesPoint, ...]]:
    """Return sparse evaluator-loss series keyed by evaluator name."""
    series_by_name: dict[str, list[OptionalFloatTimeSeriesPoint]] = {}
    for event in history:
        for evaluator_name, metrics in event.evaluators.items():
            series_by_name.setdefault(evaluator_name, []).append(
                _evaluator_loss_point(event, metrics)
            )
    return {
        evaluator_name: tuple(points)
        for evaluator_name, points in series_by_name.items()
    }


def active_evaluator_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[ActiveEvaluatorTimeSeriesPoint, ...]:
    """Return the active evaluator recorded for each cycle."""
    return tuple(
        ActiveEvaluatorTimeSeriesPoint(
            cycle_index=event.cycle_index,
            generation=event.generation,
            timestamp_utc=event.timestamp_utc,
            active_evaluator_name=_active_evaluator_name_from_event(event),
        )
        for event in history
    )


def summarize_evaluator_selection(
    history: Sequence[MorpionBootstrapEvent],
) -> EvaluatorSelectionSummary:
    """Summarize active-evaluator usage and switching across history."""
    active_names = [
        point.active_evaluator_name
        for point in active_evaluator_series(history)
        if point.active_evaluator_name is not None
    ]
    active_counts: dict[str, int] = {}
    for active_name in active_names:
        active_counts[active_name] = active_counts.get(active_name, 0) + 1

    num_switches = 0
    previous_name: str | None = None
    for active_name in active_names:
        if previous_name is not None and active_name != previous_name:
            num_switches += 1
        previous_name = active_name

    return EvaluatorSelectionSummary(
        latest_active_evaluator_name=None if not active_names else active_names[-1],
        num_switches=num_switches,
        active_counts=active_counts,
    )


def summarize_record_progress(
    history: Sequence[MorpionBootstrapEvent],
) -> MorpionRecordProgressSummary:
    """Summarize canonical Morpion record progression across history."""
    score_series = canonical_record_score_series(history)
    total_points_series = record_total_points_series(history)

    latest_score = None if not score_series else score_series[-1].value
    best_score = max(
        (point.value for point in score_series if point.value is not None),
        default=None,
    )
    if best_score is None:
        first_cycle_reaching_best = None
    else:
        first_cycle_reaching_best = next(
            point.cycle_index
            for point in score_series
            if point.value == best_score
        )

    latest_total_points = None if not total_points_series else total_points_series[-1].value
    best_total_points = max(
        (point.value for point in total_points_series if point.value is not None),
        default=None,
    )

    return MorpionRecordProgressSummary(
        latest_score=latest_score,
        best_score=best_score,
        first_cycle_reaching_best=first_cycle_reaching_best,
        latest_total_points=latest_total_points,
        best_total_points=best_total_points,
    )


def build_morpion_bootstrap_dashboard_data(
    work_dir: str | Path,
) -> MorpionBootstrapDashboardData:
    """Load one bootstrap run and build the dashboard-ready data payload."""
    run_view = load_morpion_bootstrap_run_view(work_dir)
    history = run_view.history
    return MorpionBootstrapDashboardData(
        run_summary=summarize_bootstrap_run(run_view),
        evaluator_selection_summary=summarize_evaluator_selection(history),
        record_progress_summary=summarize_record_progress(history),
        tree_num_nodes=tree_num_nodes_series(history),
        canonical_record_score=canonical_record_score_series(history),
        record_total_points=record_total_points_series(history),
        dataset_num_rows=dataset_num_rows_series(history),
        evaluator_loss_by_name=evaluator_loss_series_by_name(history),
        active_evaluator=active_evaluator_series(history),
    )


def _latest_known_event(
    run_view: MorpionBootstrapRunView,
) -> MorpionBootstrapEvent | None:
    """Return the latest available event from history or latest-status fallback."""
    if run_view.history:
        return run_view.history[-1]
    return run_view.latest_status.latest_event


def _latest_cycle_index(run_view: MorpionBootstrapRunView) -> int | None:
    """Return the latest known cycle index for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.cycle_index
    if run_view.latest_status.latest_cycle_index is not None:
        return run_view.latest_status.latest_cycle_index
    if run_view.run_state is not None:
        return run_view.run_state.cycle_index
    return None


def _latest_generation(run_view: MorpionBootstrapRunView) -> int | None:
    """Return the latest known generation for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.generation
    if run_view.latest_status.latest_generation is not None:
        return run_view.latest_status.latest_generation
    if run_view.run_state is not None:
        return run_view.run_state.generation
    return None


def _latest_tree_num_nodes(run_view: MorpionBootstrapRunView) -> int | None:
    """Return the latest known tree size for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.tree.num_nodes
    if run_view.run_state is not None:
        return run_view.run_state.tree_size_at_last_save
    return None


def _latest_active_evaluator_name(run_view: MorpionBootstrapRunView) -> str | None:
    """Return the latest known active evaluator name for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        active_name = _active_evaluator_name_from_event(latest_event)
        if active_name is not None:
            return active_name
    if run_view.run_state is not None:
        return run_view.run_state.active_evaluator_name
    return None


def _active_evaluator_name_from_event(
    event: MorpionBootstrapEvent,
) -> str | None:
    """Extract the active evaluator name from event metadata when present."""
    raw_value = event.metadata.get("active_evaluator_name")
    return raw_value if isinstance(raw_value, str) else None


def _evaluator_loss_point(
    event: MorpionBootstrapEvent,
    metrics: MorpionEvaluatorMetrics,
) -> OptionalFloatTimeSeriesPoint:
    """Build one evaluator-loss point from one event and metrics payload."""
    return OptionalFloatTimeSeriesPoint(
        cycle_index=event.cycle_index,
        generation=event.generation,
        timestamp_utc=event.timestamp_utc,
        value=metrics.final_loss,
    )


__all__ = [
    "ActiveEvaluatorTimeSeriesPoint",
    "EvaluatorSelectionSummary",
    "IntTimeSeriesPoint",
    "MorpionBootstrapDashboardData",
    "MorpionBootstrapRunSummary",
    "MorpionBootstrapRunView",
    "MorpionRecordProgressSummary",
    "OptionalFloatTimeSeriesPoint",
    "OptionalIntTimeSeriesPoint",
    "TrainingTriggeredTimeSeriesPoint",
    "active_evaluator_series",
    "build_morpion_bootstrap_dashboard_data",
    "canonical_record_score_series",
    "dataset_num_rows_series",
    "evaluator_loss_series_by_name",
    "load_morpion_bootstrap_run_view",
    "record_total_points_series",
    "summarize_bootstrap_run",
    "summarize_evaluator_selection",
    "summarize_record_progress",
    "training_triggered_series",
    "tree_num_nodes_series",
]
