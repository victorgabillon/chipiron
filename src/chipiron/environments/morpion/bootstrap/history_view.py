"""Dashboard-ready read-only history views for Morpion bootstrap runs."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from atomheart.games.morpion.state import MorpionState as AtomMorpionState

from anemone.training_export import load_training_tree_snapshot

from chipiron.displays.morpion_svg_adapter import MorpionSvgAdapter
from chipiron.environments.morpion.learning import (
    InvalidMorpionStateRefPayloadError,
    decode_morpion_state_ref_payload,
)
from chipiron.environments.morpion.morpion_display import (
    MorpionNumberedPointReplayError,
    build_morpion_display_payload,
)
from chipiron.environments.morpion.types import MorpionDynamics

from .bootstrap_loop import MorpionBootstrapPaths
from .history import (
    MorpionBootstrapEvent,
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapLatestStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
    load_bootstrap_history,
    load_latest_bootstrap_status,
)
from .record_status import (
    MorpionBootstrapRecordStatus,
    current_frontier_score,
    current_record_score,
    select_best_certified_record_candidate_from_training_tree_snapshot,
)
from .run_state import MorpionBootstrapRunState, load_bootstrap_run_state

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRunView:
    """Loaded bootstrap artifacts for one Morpion work directory."""

    work_dir: Path
    run_state: MorpionBootstrapRunState | None
    latest_status: MorpionBootstrapLatestStatus
    history: tuple[MorpionBootstrapEvent, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedTreeSnapshotReference:
    """Resolved latest tree snapshot path and optional fallback status message."""

    snapshot_path: Path | None
    snapshot_source: str | None
    status_message: str | None = None


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
    latest_frontier_score: int | None
    latest_frontier_total_points: int | None
    latest_frontier_source: str | None
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
class TreeDepthDistributionRow:
    """One compact row describing the saved tree shape at one depth."""

    depth: int
    num_nodes: int
    cumulative_nodes: int


@dataclass(frozen=True, slots=True)
class DiskUsageRow:
    """One dashboard-friendly disk-usage breakdown row."""

    label: str
    num_bytes: int | None


@dataclass(frozen=True, slots=True)
class DiskUsageSummary:
    """Disk-usage summary for one Morpion bootstrap work directory."""

    run_dir_num_bytes: int | None
    device_free_num_bytes: int | None
    device_used_num_bytes: int | None
    device_total_num_bytes: int | None
    run_dir_pct_of_device_total: float | None
    breakdown_rows: tuple[DiskUsageRow, ...]


@dataclass(frozen=True, slots=True)
class MorpionBootstrapDashboardData:
    """Bundled summaries and time series for future dashboard consumption."""

    run_summary: MorpionBootstrapRunSummary
    disk_usage_summary: DiskUsageSummary
    latest_tree_snapshot_status_message: str | None
    latest_tree_status: MorpionBootstrapTreeStatus | None
    latest_certified_record_status: MorpionBootstrapRecordStatus | None
    latest_frontier_status: MorpionBootstrapFrontierStatus | None
    evaluator_selection_summary: EvaluatorSelectionSummary
    record_progress_summary: MorpionRecordProgressSummary
    tree_num_nodes: tuple[IntTimeSeriesPoint, ...]
    canonical_record_score: tuple[OptionalIntTimeSeriesPoint, ...]
    certified_record_score: tuple[OptionalIntTimeSeriesPoint, ...]
    record_total_points: tuple[OptionalIntTimeSeriesPoint, ...]
    dataset_num_rows: tuple[OptionalIntTimeSeriesPoint, ...]
    evaluator_loss_by_name: Mapping[str, tuple[OptionalFloatTimeSeriesPoint, ...]]
    active_evaluator: tuple[ActiveEvaluatorTimeSeriesPoint, ...]
    latest_tree_depth_distribution: tuple[TreeDepthDistributionRow, ...]


@dataclass(frozen=True, slots=True)
class MorpionBootstrapCertifiedRecordBoardView:
    """Dashboard-ready rendering of the current strict certified Morpion record."""

    variant: str
    moves_since_start: int
    total_points: int
    is_exact: bool
    is_terminal: bool
    source: str
    board_svg: str
    board_text: str | None


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
    latest_record_status = None if latest_event is None else latest_event.record
    if latest_record_status is None and latest_run_state is not None:
        latest_record_status = latest_run_state.latest_record_status
    latest_frontier_status = None if latest_event is None else latest_event.frontier
    if latest_frontier_status is None and latest_run_state is not None:
        latest_frontier_status = latest_run_state.latest_frontier_status

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
        latest_frontier_score=None
        if latest_frontier_status is None
        else current_frontier_score(latest_frontier_status),
        latest_frontier_total_points=None
        if latest_frontier_status is None
        else latest_frontier_status.current_best_total_points,
        latest_frontier_source=None
        if latest_frontier_status is None
        else latest_frontier_status.current_best_source,
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


def certified_record_score_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[OptionalIntTimeSeriesPoint, ...]:
    """Return the strict certified-record series keyed by cycle."""
    return canonical_record_score_series(history)


def certified_record_best_so_far_series(
    history: Sequence[MorpionBootstrapEvent],
) -> tuple[OptionalIntTimeSeriesPoint, ...]:
    """Return best-so-far certified record points keyed by event timestamp."""
    best_score: int | None = None
    points: list[OptionalIntTimeSeriesPoint] = []
    for event in history:
        current_score = current_record_score(event.record)
        if current_score is not None and (
            best_score is None or current_score > best_score
        ):
            best_score = current_score
        points.append(
            OptionalIntTimeSeriesPoint(
                cycle_index=event.cycle_index,
                generation=event.generation,
                timestamp_utc=event.timestamp_utc,
                value=best_score,
            )
        )
    return tuple(points)


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
    resolved_tree_snapshot = _resolve_latest_tree_snapshot_reference(run_view)
    return MorpionBootstrapDashboardData(
        run_summary=summarize_bootstrap_run(run_view),
        disk_usage_summary=build_disk_usage_summary(run_view.work_dir),
        latest_tree_snapshot_status_message=resolved_tree_snapshot.status_message,
        latest_tree_status=_latest_tree_status(run_view),
        latest_certified_record_status=_latest_certified_record_status(run_view),
        latest_frontier_status=_latest_frontier_status(run_view),
        evaluator_selection_summary=summarize_evaluator_selection(history),
        record_progress_summary=summarize_record_progress(history),
        tree_num_nodes=tree_num_nodes_series(history),
        canonical_record_score=canonical_record_score_series(history),
        certified_record_score=certified_record_best_so_far_series(history),
        record_total_points=record_total_points_series(history),
        dataset_num_rows=dataset_num_rows_series(history),
        evaluator_loss_by_name=evaluator_loss_series_by_name(history),
        active_evaluator=active_evaluator_series(history),
        latest_tree_depth_distribution=latest_tree_depth_distribution(run_view),
    )


def build_current_certified_record_board_view(
    work_dir: str | Path,
) -> MorpionBootstrapCertifiedRecordBoardView | None:
    """Build the current strict certified-record board view for the dashboard."""
    run_view = load_morpion_bootstrap_run_view(work_dir)
    resolved_snapshot = _resolve_latest_tree_snapshot_reference(run_view)
    snapshot_path = resolved_snapshot.snapshot_path
    if snapshot_path is None:
        return None

    try:
        snapshot = load_training_tree_snapshot(snapshot_path)
    except OSError:
        LOGGER.exception(
            "[dashboard] certified_record_board_snapshot_load_failed path=%s",
            str(snapshot_path),
        )
        return None

    candidate = select_best_certified_record_candidate_from_training_tree_snapshot(
        snapshot
    )
    if candidate is None:
        return None

    try:
        atom_state = decode_morpion_state_ref_payload(candidate.state_ref_payload)
        return _certified_record_board_view_from_atom_state(
            atom_state=atom_state,
            state_ref_payload=candidate.state_ref_payload,
            is_exact=True,
            is_terminal=True,
            source="certified_terminal_leaf",
        )
    except (
        InvalidMorpionStateRefPayloadError,
        MorpionNumberedPointReplayError,
        ValueError,
        TypeError,
    ):
        LOGGER.exception(
            "[dashboard] certified_record_board_reconstruction_failed path=%s node_id=%s",
            str(snapshot_path),
            candidate.node_id,
        )
        return None


def format_num_bytes(num_bytes: int | None) -> str:
    """Format a byte count into one compact human-readable size string."""
    if num_bytes is None:
        return "unknown"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ("KB", "MB", "GB", "TB", "PB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"


def _certified_record_board_view_from_atom_state(
    *,
    atom_state: AtomMorpionState,
    state_ref_payload: Mapping[str, object],
    is_exact: bool,
    is_terminal: bool,
    source: str,
) -> MorpionBootstrapCertifiedRecordBoardView:
    """Render one decoded certified Morpion state through the shared SVG pipeline."""
    dynamics = MorpionDynamics()
    state = dynamics.wrap_atomheart_state(atom_state)
    svg_adapter = MorpionSvgAdapter()
    position = svg_adapter.position_from_update(
        state_tag=state.tag,
        adapter_payload=build_morpion_display_payload(
            state=state,
            dynamics=dynamics,
            state_ref_payload=state_ref_payload,
        ),
    )
    render_result = svg_adapter.render_svg(position, 720, margin=8)
    return MorpionBootstrapCertifiedRecordBoardView(
        variant=state.variant.value,
        moves_since_start=state.moves,
        total_points=len(state.points),
        is_exact=is_exact,
        is_terminal=is_terminal,
        source=source,
        board_svg=render_result.svg_bytes.decode("utf-8"),
        board_text=state.pprint(),
    )


def recursive_path_num_bytes(path: Path) -> int | None:
    """Return the recursive size of one path, treating missing paths as zero."""
    if not path.exists():
        return 0
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return None
    return total


def filesystem_usage_for_path(path: Path) -> tuple[int, int, int] | None:
    """Return ``(free, used, total)`` for the device containing ``path``."""
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return None
    return (usage.free, usage.used, usage.total)


def build_disk_usage_summary(work_dir: str | Path) -> DiskUsageSummary:
    """Build one operator-friendly disk-usage summary for the bootstrap dashboard."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    run_dir_num_bytes = recursive_path_num_bytes(paths.work_dir)
    device_usage = filesystem_usage_for_path(paths.work_dir)
    if device_usage is None:
        device_free_num_bytes = None
        device_used_num_bytes = None
        device_total_num_bytes = None
        run_dir_pct_of_device_total = None
    else:
        device_free_num_bytes, device_used_num_bytes, device_total_num_bytes = device_usage
        run_dir_pct_of_device_total = (
            None
            if run_dir_num_bytes is None or device_total_num_bytes <= 0
            else (run_dir_num_bytes / device_total_num_bytes) * 100.0
        )

    breakdown_candidates = (
        DiskUsageRow("models", recursive_path_num_bytes(paths.model_dir)),
        DiskUsageRow("rows", recursive_path_num_bytes(paths.rows_dir)),
        DiskUsageRow(
            "search_checkpoints",
            recursive_path_num_bytes(paths.runtime_checkpoint_dir),
        ),
        DiskUsageRow("tree_exports", recursive_path_num_bytes(paths.tree_snapshot_dir)),
        DiskUsageRow(
            "history_logs_status",
            _combined_file_group_num_bytes(
                (
                    paths.history_jsonl_path,
                    paths.latest_status_path,
                    paths.run_state_path,
                    paths.bootstrap_config_path,
                    paths.control_path,
                    paths.launcher_pid_path,
                    paths.launcher_process_state_path,
                    paths.launcher_stdout_log_path,
                    paths.launcher_stderr_log_path,
                )
            ),
        ),
    )
    breakdown_rows = tuple(
        sorted(
            breakdown_candidates,
            key=lambda row: (
                row.num_bytes is None,
                0 if row.num_bytes is None else -row.num_bytes,
                row.label,
            ),
        )
    )
    return DiskUsageSummary(
        run_dir_num_bytes=run_dir_num_bytes,
        device_free_num_bytes=device_free_num_bytes,
        device_used_num_bytes=device_used_num_bytes,
        device_total_num_bytes=device_total_num_bytes,
        run_dir_pct_of_device_total=run_dir_pct_of_device_total,
        breakdown_rows=breakdown_rows,
    )


def _combined_file_group_num_bytes(paths: Sequence[Path]) -> int | None:
    """Return the summed size for one group of file paths, missing files as zero."""
    total = 0
    for path in paths:
        num_bytes = recursive_path_num_bytes(path)
        if num_bytes is None:
            return None
        total += num_bytes
    return total


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
    resolved_snapshot = _resolve_latest_tree_snapshot_reference(run_view)
    if resolved_snapshot.snapshot_path is not None:
        snapshot = load_training_tree_snapshot(resolved_snapshot.snapshot_path)
        return len(snapshot.nodes)
    if run_view.run_state is not None:
        return run_view.run_state.tree_size_at_last_save
    return None


def _latest_tree_status(
    run_view: MorpionBootstrapRunView,
) -> MorpionBootstrapTreeStatus | None:
    """Return the latest known tree-structure status for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.tree
    resolved_snapshot = _resolve_latest_tree_snapshot_reference(run_view)
    if resolved_snapshot.snapshot_path is not None:
        snapshot = load_training_tree_snapshot(resolved_snapshot.snapshot_path)
        depth_counts: dict[int, int] = {}
        for node in snapshot.nodes:
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        return MorpionBootstrapTreeStatus(
            num_nodes=len(snapshot.nodes),
            min_depth_present=None if not depth_counts else min(depth_counts),
            max_depth_present=None if not depth_counts else max(depth_counts),
            depth_node_counts=depth_counts,
        )
    if run_view.run_state is not None:
        return MorpionBootstrapTreeStatus(
            num_nodes=run_view.run_state.tree_size_at_last_save
        )
    return None


def latest_tree_depth_distribution(
    run_view: MorpionBootstrapRunView,
) -> tuple[TreeDepthDistributionRow, ...]:
    """Return the latest persisted tree depth distribution for the dashboard."""
    latest_tree_status = _latest_tree_status(run_view)
    if latest_tree_status is not None and latest_tree_status.depth_node_counts:
        return _tree_depth_distribution_rows_from_counts(
            latest_tree_status.depth_node_counts
        )

    resolved_snapshot = _resolve_latest_tree_snapshot_reference(run_view)
    snapshot_path = resolved_snapshot.snapshot_path
    if snapshot_path is None:
        return ()
    snapshot = load_training_tree_snapshot(snapshot_path)
    depth_counts: dict[int, int] = {}
    for node in snapshot.nodes:
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
    return _tree_depth_distribution_rows_from_counts(depth_counts)


def _latest_certified_record_status(
    run_view: MorpionBootstrapRunView,
) -> MorpionBootstrapRecordStatus | None:
    """Return the latest known certified record status for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.record
    if run_view.run_state is not None:
        return run_view.run_state.latest_record_status
    return None


def _latest_frontier_status(
    run_view: MorpionBootstrapRunView,
) -> MorpionBootstrapFrontierStatus | None:
    """Return the latest known frontier/debug status for one run view."""
    latest_event = _latest_known_event(run_view)
    if latest_event is not None:
        return latest_event.frontier
    if run_view.run_state is not None:
        return run_view.run_state.latest_frontier_status
    return None


def _latest_tree_snapshot_path(
    run_view: MorpionBootstrapRunView,
) -> Path | None:
    """Return the latest persisted tree snapshot path for one run view."""
    return _resolve_latest_tree_snapshot_reference(run_view).snapshot_path


def _resolve_latest_tree_snapshot_reference(
    run_view: MorpionBootstrapRunView,
) -> _ResolvedTreeSnapshotReference:
    """Resolve the latest usable tree snapshot path for one run view."""
    latest_event = _latest_known_event(run_view)
    persisted_path = None
    if latest_event is not None:
        persisted_path = latest_event.artifacts.tree_snapshot_path
    if persisted_path is None and run_view.run_state is not None:
        persisted_path = run_view.run_state.latest_tree_snapshot_path
    paths = MorpionBootstrapPaths.from_work_dir(run_view.work_dir)
    if persisted_path is not None:
        resolved_path = paths.resolve_work_dir_path(persisted_path)
        if resolved_path is not None and resolved_path.is_file():
            return _ResolvedTreeSnapshotReference(
                snapshot_path=resolved_path,
                snapshot_source="metadata",
            )
        latest_on_disk = _latest_generation_json_path(paths.tree_snapshot_dir)
        if latest_on_disk is not None:
            status_message = (
                "Tree snapshot metadata points to a missing file; using the newest "
                "tree export discovered on disk instead."
            )
            LOGGER.warning(
                "[dashboard] latest_tree_snapshot_missing metadata_path=%s fallback_path=%s",
                persisted_path,
                str(latest_on_disk),
            )
            return _ResolvedTreeSnapshotReference(
                snapshot_path=latest_on_disk,
                snapshot_source="tree_snapshot_dir",
                status_message=status_message,
            )
        return _ResolvedTreeSnapshotReference(
            snapshot_path=None,
            snapshot_source=None,
            status_message=(
                "Tree snapshot metadata points to a missing file, and no fallback "
                "tree export exists on disk."
            ),
        )
    latest_on_disk = _latest_generation_json_path(paths.tree_snapshot_dir)
    if latest_on_disk is None:
        return _ResolvedTreeSnapshotReference(
            snapshot_path=None,
            snapshot_source=None,
        )
    return _ResolvedTreeSnapshotReference(
        snapshot_path=latest_on_disk,
        snapshot_source="tree_snapshot_dir",
        status_message="Using the latest tree export discovered on disk.",
    )


def _latest_generation_json_path(directory: Path) -> Path | None:
    """Return the newest ``generation_*.json`` file from one directory."""
    candidates = sorted(directory.glob("generation_*.json"))
    return None if not candidates else candidates[-1]


def _tree_depth_distribution_rows_from_counts(
    depth_counts: Mapping[int, int],
) -> tuple[TreeDepthDistributionRow, ...]:
    """Return one cumulative depth-distribution table from node counts."""
    cumulative_nodes = 0
    rows: list[TreeDepthDistributionRow] = []
    for depth in sorted(depth_counts):
        cumulative_nodes += depth_counts[depth]
        rows.append(
            TreeDepthDistributionRow(
                depth=depth,
                num_nodes=depth_counts[depth],
                cumulative_nodes=cumulative_nodes,
            )
        )
    return tuple(rows)


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
    "DiskUsageRow",
    "DiskUsageSummary",
    "EvaluatorSelectionSummary",
    "IntTimeSeriesPoint",
    "MorpionBootstrapDashboardData",
    "MorpionBootstrapRunSummary",
    "MorpionBootstrapRunView",
    "MorpionRecordProgressSummary",
    "OptionalFloatTimeSeriesPoint",
    "OptionalIntTimeSeriesPoint",
    "TrainingTriggeredTimeSeriesPoint",
    "TreeDepthDistributionRow",
    "active_evaluator_series",
    "build_disk_usage_summary",
    "build_morpion_bootstrap_dashboard_data",
    "canonical_record_score_series",
    "certified_record_best_so_far_series",
    "certified_record_score_series",
    "dataset_num_rows_series",
    "evaluator_loss_series_by_name",
    "filesystem_usage_for_path",
    "format_num_bytes",
    "latest_tree_depth_distribution",
    "load_morpion_bootstrap_run_view",
    "record_total_points_series",
    "recursive_path_num_bytes",
    "summarize_bootstrap_run",
    "summarize_evaluator_selection",
    "summarize_record_progress",
    "training_triggered_series",
    "tree_num_nodes_series",
]
