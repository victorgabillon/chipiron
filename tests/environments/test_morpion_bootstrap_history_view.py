"""Tests for dashboard-ready Morpion bootstrap history views."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    save_training_tree_snapshot,
)
from atomheart.games.morpion import MorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import (
    ActiveEvaluatorTimeSeriesPoint,
    EvaluatorSelectionSummary,
    MorpionBootstrapArtifacts,
    MorpionBootstrapDashboardData,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapLatestStatus,
    MorpionBootstrapPaths,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapRunState,
    MorpionBootstrapRunView,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
    MorpionPipelineEvaluatorTrainingResult,
    MorpionRecordProgressSummary,
    TrainingTriggeredTimeSeriesPoint,
    TreeDepthDistributionRow,
    active_evaluator_series,
    build_morpion_bootstrap_dashboard_data,
    canonical_record_score_series,
    certified_record_best_so_far_series,
    dataset_num_rows_series,
    evaluator_loss_series_by_name,
    latest_tree_depth_distribution,
    load_morpion_bootstrap_run_view,
    record_total_points_series,
    save_pipeline_training_status_file,
    save_bootstrap_run_state,
    summarize_bootstrap_run,
    summarize_evaluator_selection,
    summarize_record_progress,
    training_triggered_series,
    tree_num_nodes_series,
)
from chipiron.environments.morpion.bootstrap.history_view import (
    DiskUsageRow,
    DiskUsageSummary,
    build_current_certified_record_board_view,
    build_disk_usage_summary,
    format_num_bytes,
    recursive_path_num_bytes,
)


def _make_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    training_triggered: bool,
    tree_num_nodes: int,
    record_score: int | None,
    total_points: int | None,
    active_evaluator_name: str | None,
    dataset_num_rows: int | None = None,
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] | None = None,
) -> MorpionBootstrapEvent:
    """Build one representative bootstrap event for history-view tests."""
    metadata: dict[str, object] = {
        "game": "morpion",
        "variant": "5T",
        "initial_pattern": "greek_cross",
        "initial_point_count": 36,
    }
    if active_evaluator_name is not None:
        metadata["active_evaluator_name"] = active_evaluator_name
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}",
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc=timestamp_utc,
        tree=MorpionBootstrapTreeStatus(
            num_nodes=tree_num_nodes,
            min_depth_present=0,
            max_depth_present=2,
            depth_node_counts={0: 1, 1: max(tree_num_nodes - 5, 0), 2: 4},
        ),
        dataset=MorpionBootstrapDatasetStatus(
            num_rows=dataset_num_rows,
            num_samples=dataset_num_rows,
        ),
        training=MorpionBootstrapTrainingStatus(triggered=training_triggered),
        record=MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=record_score,
            current_best_total_points=total_points,
            current_best_is_exact=True if record_score is not None else None,
            current_best_is_terminal=True if record_score is not None else None,
            current_best_source="certified_terminal_leaf"
            if record_score is not None
            else None,
        ),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=None,
            rows_path=None,
        ),
        evaluators={} if evaluator_metrics is None else dict(evaluator_metrics),
        metadata=metadata,
    )


def _make_run_state(
    *,
    tree_size_at_last_save: int = 25,
) -> MorpionBootstrapRunState:
    """Build one representative run state for run-view loading tests."""
    return MorpionBootstrapRunState(
        generation=2,
        cycle_index=3,
        latest_tree_snapshot_path="tree_exports/generation_000002.json",
        latest_rows_path="rows/generation_000002.json",
        latest_model_bundle_paths={"mlp": "models/generation_000002/mlp"},
        active_evaluator_name="mlp",
        tree_size_at_last_save=tree_size_at_last_save,
        last_save_unix_s=200.0,
        latest_record_status=MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=19,
            current_best_total_points=55,
            current_best_is_exact=True,
            current_best_is_terminal=True,
            current_best_source="certified_terminal_leaf",
        ),
    )


def _make_morpion_payload(move_count: int) -> dict[str, object]:
    """Build one real Morpion checkpoint payload after ``move_count`` moves."""
    dynamics = MorpionDynamics()
    state = morpion_initial_state()
    for _ in range(move_count):
        action = dynamics.all_legal_actions(state)[0]
        state = dynamics.step(state, action).next_state
    codec = MorpionStateCheckpointCodec()
    return codec.dump_state_ref(state)


def test_load_run_view_from_artifacts(tmp_path: Path) -> None:
    """The run-view loader should aggregate run-state, history, and latest status."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())
    first_event = _make_event(
        cycle_index=0,
        generation=1,
        timestamp_utc="2026-04-11T08:00:00Z",
        training_triggered=True,
        tree_num_nodes=10,
        record_score=12,
        total_points=48,
        active_evaluator_name="linear",
        dataset_num_rows=10,
    )
    second_event = _make_event(
        cycle_index=1,
        generation=2,
        timestamp_utc="2026-04-11T09:00:00Z",
        training_triggered=False,
        tree_num_nodes=15,
        record_score=14,
        total_points=50,
        active_evaluator_name="mlp",
        dataset_num_rows=None,
    )
    recorder.record(first_event)
    recorder.record(second_event)
    run_state = _make_run_state()
    save_bootstrap_run_state(run_state, paths.run_state_path)

    run_view = load_morpion_bootstrap_run_view(tmp_path)

    assert run_view.work_dir == paths.work_dir
    assert run_view.run_state == run_state
    assert run_view.history == (first_event, second_event)
    assert run_view.latest_status.latest_event == second_event
    assert run_view.latest_status.latest_cycle_index == 1
    assert run_view.latest_status.latest_generation == 2


def test_run_summary_falls_back_to_run_state_tree_size_without_history() -> None:
    """Run summary should use saved tree size when no event history exists."""
    run_state = _make_run_state(tree_size_at_last_save=25)
    run_view = MorpionBootstrapRunView(
        work_dir=Path("/tmp/run"),
        run_state=run_state,
        latest_status=MorpionBootstrapLatestStatus(
            work_dir="/tmp/run",
            latest_generation=None,
            latest_cycle_index=None,
            latest_event=None,
        ),
        history=(),
    )

    summary = summarize_bootstrap_run(run_view)

    assert summary.latest_cycle_index == run_state.cycle_index
    assert summary.latest_generation == run_state.generation
    assert summary.latest_active_evaluator_name == run_state.active_evaluator_name
    assert summary.latest_tree_num_nodes == 25


def test_empty_run_view_works(tmp_path: Path) -> None:
    """An empty work directory should still load as a valid run view."""
    run_view = load_morpion_bootstrap_run_view(tmp_path)

    assert run_view.work_dir == tmp_path.resolve()
    assert run_view.run_state is None
    assert run_view.history == ()
    assert run_view.latest_status.latest_event is None
    assert run_view.latest_status.latest_cycle_index is None
    assert run_view.latest_status.latest_generation is None


def test_build_current_certified_record_board_view_renders_numbered_points(
    tmp_path: Path,
) -> None:
    """Certified-record board view should render numbered post-start points."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    snapshot = TrainingTreeSnapshot(
        nodes=(
            TrainingNodeSnapshot(
                node_id="certified-2",
                parent_ids=(),
                child_ids=(),
                depth=2,
                state_ref_payload=_make_morpion_payload(2),
                direct_value_scalar=2.0,
                backed_up_value_scalar=2.0,
                is_terminal=True,
                is_exact=True,
                over_event_label=None,
                visit_count=3,
                metadata={"source": "history-view-test"},
            ),
        ),
        root_node_id="certified-2",
    )
    save_training_tree_snapshot(
        snapshot, paths.tree_snapshot_dir / "generation_000001.json"
    )

    board_view = build_current_certified_record_board_view(tmp_path)

    assert board_view is not None
    assert board_view.variant == "5T"
    assert board_view.moves_since_start == 2
    assert board_view.total_points == 38
    assert board_view.is_exact is True
    assert board_view.is_terminal is True
    assert board_view.source == "certified_terminal_leaf"
    assert ">1</text>" in board_view.board_svg
    assert ">2</text>" in board_view.board_svg
    assert "#0f766e" in board_view.board_svg
    assert board_view.board_text is not None


def test_format_num_bytes_is_human_readable() -> None:
    """Byte formatting should stay compact and operator-friendly."""
    assert format_num_bytes(None) == "unknown"
    assert format_num_bytes(12) == "12 B"
    assert format_num_bytes(1536) == "1.5 KB"
    assert format_num_bytes(1048576) == "1.0 MB"


def test_recursive_path_num_bytes_sums_directory_contents(tmp_path: Path) -> None:
    """Recursive size helper should sum nested files and treat missing paths as zero."""
    (tmp_path / "root.bin").write_bytes(b"abc")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "child.bin").write_bytes(b"defgh")

    assert recursive_path_num_bytes(tmp_path) == 8
    assert recursive_path_num_bytes(tmp_path / "missing") == 0


def test_build_disk_usage_summary_collects_expected_breakdown(tmp_path: Path) -> None:
    """Disk summary should report run totals and sort breakdown rows by size."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    (paths.model_dir / "weights.bin").write_bytes(b"abcdef")
    (paths.rows_dir / "rows.json").write_bytes(b"1234")
    (paths.runtime_checkpoint_dir / "generation_000001.json").write_bytes(b"12")
    (paths.tree_snapshot_dir / "generation_000001.json").write_bytes(b"123")
    paths.history_jsonl_path.write_bytes(b"12345")
    paths.latest_status_path.write_bytes(b"12")

    summary = build_disk_usage_summary(tmp_path)

    assert isinstance(summary, DiskUsageSummary)
    assert summary.run_dir_num_bytes == 22
    assert summary.device_total_num_bytes is None or summary.device_total_num_bytes > 0
    assert (
        summary.run_dir_pct_of_device_total is None
        or summary.run_dir_pct_of_device_total >= 0.0
    )
    assert summary.breakdown_rows[:5] == (
        DiskUsageRow("history_logs_status", 7),
        DiskUsageRow("models", 6),
        DiskUsageRow("rows", 4),
        DiskUsageRow("tree_exports", 3),
        DiskUsageRow("search_checkpoints", 2),
    )


def test_build_disk_usage_summary_handles_missing_directories(tmp_path: Path) -> None:
    """Disk summary should treat absent artifact folders as zero-sized."""
    summary = build_disk_usage_summary(tmp_path)

    assert summary.run_dir_num_bytes == 0
    assert summary.breakdown_rows == (
        DiskUsageRow("history_logs_status", 0),
        DiskUsageRow("models", 0),
        DiskUsageRow("rows", 0),
        DiskUsageRow("search_checkpoints", 0),
        DiskUsageRow("tree_exports", 0),
    )


def test_dashboard_data_includes_disk_usage_summary(tmp_path: Path) -> None:
    """Dashboard data payload should include the derived disk-usage summary."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    (paths.rows_dir / "generation_000001.json").write_bytes(b"123")

    data = build_morpion_bootstrap_dashboard_data(tmp_path)

    assert data.disk_usage_summary.run_dir_num_bytes == 3
    assert any(
        row == DiskUsageRow("rows", 3) for row in data.disk_usage_summary.breakdown_rows
    )


def test_run_summary_counts_train_vs_no_train_cycles() -> None:
    """Run summary should report train and no-train counts separately."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
            dataset_num_rows=10,
        ),
        _make_event(
            cycle_index=1,
            generation=1,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=False,
            tree_num_nodes=12,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=2,
            generation=2,
            timestamp_utc="2026-04-11T10:00:00Z",
            training_triggered=True,
            tree_num_nodes=15,
            record_score=12,
            total_points=48,
            active_evaluator_name="mlp",
            dataset_num_rows=12,
        ),
    )
    run_view = MorpionBootstrapRunView(
        work_dir=Path("/tmp/run"),
        run_state=None,
        latest_status=MorpionBootstrapLatestStatus(
            work_dir="/tmp/run",
            latest_generation=None,
            latest_cycle_index=None,
            latest_event=None,
        ),
        history=history,
    )

    summary = summarize_bootstrap_run(run_view)

    assert summary.num_cycles == 3
    assert summary.num_train_cycles == 2
    assert summary.num_no_train_cycles == 1
    assert summary.latest_cycle_index == 2
    assert summary.latest_generation == 2
    assert summary.latest_active_evaluator_name == "mlp"
    assert summary.latest_tree_num_nodes == 15
    assert summary.latest_record_score == 12
    assert summary.latest_record_total_points == 48
    assert summary.latest_timestamp_utc == "2026-04-11T10:00:00Z"


def test_canonical_record_score_series_uses_moves_since_start() -> None:
    """Canonical record series must expose moves-since-start, not total points."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=18,
            total_points=54,
            active_evaluator_name="linear",
        ),
    )

    score_series = canonical_record_score_series(history)
    total_points = record_total_points_series(history)

    assert score_series[0].value == 18
    assert total_points[0].value == 54


def test_certified_record_best_so_far_series_stays_flat_between_improvements() -> None:
    """Certified record progress should be plotted as best-so-far over time."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=None,
            total_points=None,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=True,
            tree_num_nodes=12,
            record_score=12,
            total_points=48,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=2,
            generation=3,
            timestamp_utc="2026-04-11T10:00:00Z",
            training_triggered=True,
            tree_num_nodes=14,
            record_score=12,
            total_points=48,
            active_evaluator_name="mlp",
        ),
        _make_event(
            cycle_index=3,
            generation=4,
            timestamp_utc="2026-04-11T11:00:00Z",
            training_triggered=True,
            tree_num_nodes=16,
            record_score=14,
            total_points=50,
            active_evaluator_name="mlp",
        ),
    )

    assert tuple(
        point.value for point in certified_record_best_so_far_series(history)
    ) == (None, 12, 12, 14)


def test_certified_record_best_so_far_series_handles_no_certified_record() -> None:
    """Runs with no certified record yet should keep the series empty-valued."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=None,
            total_points=None,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=False,
            tree_num_nodes=11,
            record_score=None,
            total_points=None,
            active_evaluator_name="linear",
        ),
    )

    assert tuple(
        point.value for point in certified_record_best_so_far_series(history)
    ) == (None, None)


def test_evaluator_loss_series_groups_by_evaluator_name() -> None:
    """Evaluator loss series should stay sparse and grouped by evaluator name."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
            evaluator_metrics={
                "linear": MorpionEvaluatorMetrics(
                    final_loss=0.5,
                    num_epochs=1,
                    num_samples=10,
                )
            },
        ),
        _make_event(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=True,
            tree_num_nodes=12,
            record_score=12,
            total_points=48,
            active_evaluator_name="mlp",
            evaluator_metrics={
                "mlp": MorpionEvaluatorMetrics(
                    final_loss=0.3,
                    num_epochs=1,
                    num_samples=12,
                )
            },
        ),
        _make_event(
            cycle_index=2,
            generation=3,
            timestamp_utc="2026-04-11T10:00:00Z",
            training_triggered=True,
            tree_num_nodes=14,
            record_score=14,
            total_points=50,
            active_evaluator_name="mlp",
            evaluator_metrics={
                "linear": MorpionEvaluatorMetrics(
                    final_loss=0.25,
                    num_epochs=1,
                    num_samples=14,
                ),
                "mlp": MorpionEvaluatorMetrics(
                    final_loss=0.2,
                    num_epochs=1,
                    num_samples=14,
                ),
            },
        ),
    )

    loss_by_name = evaluator_loss_series_by_name(history)

    assert set(loss_by_name) == {"linear", "mlp"}
    assert tuple(point.cycle_index for point in loss_by_name["linear"]) == (0, 2)
    assert tuple(point.value for point in loss_by_name["linear"]) == (0.5, 0.25)
    assert tuple(point.cycle_index for point in loss_by_name["mlp"]) == (1, 2)
    assert tuple(point.value for point in loss_by_name["mlp"]) == (0.3, 0.2)


def test_active_evaluator_series_reads_event_metadata() -> None:
    """Active evaluator series should come from event metadata."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=True,
            tree_num_nodes=12,
            record_score=12,
            total_points=48,
            active_evaluator_name="mlp",
        ),
    )

    assert active_evaluator_series(history) == (
        ActiveEvaluatorTimeSeriesPoint(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            active_evaluator_name="linear",
        ),
        ActiveEvaluatorTimeSeriesPoint(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            active_evaluator_name="mlp",
        ),
    )


def test_evaluator_switch_summary_counts_changes_correctly() -> None:
    """Switch summary should count changes between consecutive non-null names."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=1,
            generation=1,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=False,
            tree_num_nodes=11,
            record_score=10,
            total_points=46,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=2,
            generation=2,
            timestamp_utc="2026-04-11T10:00:00Z",
            training_triggered=True,
            tree_num_nodes=12,
            record_score=11,
            total_points=47,
            active_evaluator_name="mlp",
        ),
        _make_event(
            cycle_index=3,
            generation=2,
            timestamp_utc="2026-04-11T11:00:00Z",
            training_triggered=False,
            tree_num_nodes=13,
            record_score=11,
            total_points=47,
            active_evaluator_name="mlp",
        ),
        _make_event(
            cycle_index=4,
            generation=3,
            timestamp_utc="2026-04-11T12:00:00Z",
            training_triggered=True,
            tree_num_nodes=14,
            record_score=12,
            total_points=48,
            active_evaluator_name="linear",
        ),
    )

    assert summarize_evaluator_selection(history) == EvaluatorSelectionSummary(
        latest_active_evaluator_name="linear",
        num_switches=2,
        active_counts={"linear": 3, "mlp": 2},
    )


def test_record_progress_summary_uses_canonical_score() -> None:
    """Record-progress summary must treat moves-since-start as the canonical score."""
    history = (
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            training_triggered=True,
            tree_num_nodes=10,
            record_score=18,
            total_points=54,
            active_evaluator_name="linear",
        ),
        _make_event(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            training_triggered=True,
            tree_num_nodes=12,
            record_score=20,
            total_points=56,
            active_evaluator_name="mlp",
        ),
    )

    assert summarize_record_progress(history) == MorpionRecordProgressSummary(
        latest_score=20,
        best_score=20,
        first_cycle_reaching_best=1,
        latest_total_points=56,
        best_total_points=56,
    )


def test_dashboard_data_bundles_everything(tmp_path: Path) -> None:
    """Dashboard payload builder should bundle summaries and all expected series."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())
    first_event = _make_event(
        cycle_index=0,
        generation=1,
        timestamp_utc="2026-04-11T08:00:00Z",
        training_triggered=True,
        tree_num_nodes=10,
        record_score=12,
        total_points=48,
        active_evaluator_name="linear",
        dataset_num_rows=10,
        evaluator_metrics={
            "linear": MorpionEvaluatorMetrics(
                final_loss=0.5,
                num_epochs=1,
                num_samples=10,
            )
        },
    )
    second_event = _make_event(
        cycle_index=1,
        generation=2,
        timestamp_utc="2026-04-11T09:00:00Z",
        training_triggered=False,
        tree_num_nodes=15,
        record_score=14,
        total_points=50,
        active_evaluator_name="mlp",
        dataset_num_rows=None,
    )
    recorder.record(first_event)
    recorder.record(second_event)
    save_bootstrap_run_state(_make_run_state(), paths.run_state_path)
    save_pipeline_training_status_file(
        generation=1,
        training_status="done",
        updated_at_utc="2026-04-11T08:00:00Z",
        metadata={"source": "test"},
        selected_evaluator_name="linear",
        selection_policy="lowest_final_loss",
        evaluator_results={
            "linear": MorpionPipelineEvaluatorTrainingResult(
                final_loss=0.5,
                elapsed_s=1.0,
                model_bundle_path="models/generation_000001/linear",
            )
        },
        path=paths.pipeline_training_status_path_for_generation(1),
    )

    dashboard_data = build_morpion_bootstrap_dashboard_data(tmp_path)

    assert isinstance(dashboard_data, MorpionBootstrapDashboardData)
    assert dashboard_data.run_summary.num_cycles == 2
    assert dashboard_data.latest_tree_status is not None
    assert dashboard_data.latest_tree_status.max_depth_present == 2
    assert dashboard_data.run_summary.latest_record_score == 14
    assert (
        dashboard_data.evaluator_selection_summary.latest_active_evaluator_name == "mlp"
    )
    assert dashboard_data.record_progress_summary.best_score == 14
    assert len(dashboard_data.tree_num_nodes) == 2
    assert len(dashboard_data.canonical_record_score) == 2
    assert tuple(point.value for point in dashboard_data.certified_record_score) == (
        12,
        14,
    )
    assert len(dashboard_data.record_total_points) == 2
    assert len(dashboard_data.dataset_num_rows) == 2
    assert set(dashboard_data.evaluator_loss_by_name) == {"linear"}
    assert len(dashboard_data.active_evaluator) == 2
    assert dashboard_data.latest_tree_depth_distribution == (
        TreeDepthDistributionRow(depth=0, num_nodes=1, cumulative_nodes=1),
        TreeDepthDistributionRow(depth=1, num_nodes=10, cumulative_nodes=11),
        TreeDepthDistributionRow(depth=2, num_nodes=4, cumulative_nodes=15),
    )
    assert training_triggered_series((first_event, second_event)) == (
        TrainingTriggeredTimeSeriesPoint(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            triggered=True,
        ),
        TrainingTriggeredTimeSeriesPoint(
            cycle_index=1,
            generation=2,
            timestamp_utc="2026-04-11T09:00:00Z",
            triggered=False,
        ),
    )
    assert tuple(
        point.value for point in tree_num_nodes_series((first_event, second_event))
    ) == (
        10,
        15,
    )
    assert tuple(
        point.value for point in dataset_num_rows_series((first_event, second_event))
    ) == (
        10,
        None,
    )


def test_dashboard_data_tolerates_old_training_status_without_evaluator_results(
    tmp_path: Path,
) -> None:
    """Dashboard data loading should skip old training status payloads safely."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.pipeline_training_status_path_for_generation(1).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.pipeline_training_status_path_for_generation(1).write_text(
        json.dumps(
            {
                "generation": 1,
                "status": "done",
                "updated_at_utc": "2026-04-11T08:00:00Z",
                "metadata": {"source": "old-format"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_bootstrap_run_state(_make_run_state(), paths.run_state_path)

    dashboard_data = build_morpion_bootstrap_dashboard_data(tmp_path)

    assert dashboard_data.evaluator_loss_by_name == {}

def test_latest_tree_depth_distribution_falls_back_to_snapshot(tmp_path: Path) -> None:
    """Depth distribution should fall back to the latest saved tree snapshot."""
    snapshot_path = (
        MorpionBootstrapPaths.from_work_dir(tmp_path).tree_snapshot_dir
        / "generation_000001.json"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_tree_snapshot(
        TrainingTreeSnapshot(
            root_node_id="root",
            nodes=(
                TrainingNodeSnapshot(
                    node_id="root",
                    parent_ids=(),
                    child_ids=("child-1", "child-2"),
                    depth=0,
                    state_ref_payload={"kind": "root"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=False,
                    is_exact=False,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
                TrainingNodeSnapshot(
                    node_id="child-1",
                    parent_ids=("root",),
                    child_ids=("leaf",),
                    depth=1,
                    state_ref_payload={"kind": "child-1"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=False,
                    is_exact=False,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
                TrainingNodeSnapshot(
                    node_id="child-2",
                    parent_ids=("root",),
                    child_ids=(),
                    depth=1,
                    state_ref_payload={"kind": "child-2"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=False,
                    is_exact=False,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
                TrainingNodeSnapshot(
                    node_id="leaf",
                    parent_ids=("child-1",),
                    child_ids=(),
                    depth=2,
                    state_ref_payload={"kind": "leaf"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=True,
                    is_exact=True,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
            ),
            metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
        ),
        snapshot_path,
    )
    run_view = MorpionBootstrapRunView(
        work_dir=tmp_path,
        run_state=MorpionBootstrapRunState(
            generation=1,
            cycle_index=0,
            latest_tree_snapshot_path="tree_exports/generation_000001.json",
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=4,
            last_save_unix_s=None,
        ),
        latest_status=MorpionBootstrapLatestStatus(
            work_dir=str(tmp_path),
            latest_generation=None,
            latest_cycle_index=None,
            latest_event=None,
        ),
        history=(
            MorpionBootstrapEvent(
                cycle_index=0,
                generation=1,
                timestamp_utc="2026-04-11T08:00:00Z",
                tree=MorpionBootstrapTreeStatus(num_nodes=4),
                dataset=MorpionBootstrapDatasetStatus(num_rows=None, num_samples=None),
                training=MorpionBootstrapTrainingStatus(triggered=False),
            ),
        ),
    )

    assert latest_tree_depth_distribution(run_view) == (
        TreeDepthDistributionRow(depth=0, num_nodes=1, cumulative_nodes=1),
        TreeDepthDistributionRow(depth=1, num_nodes=2, cumulative_nodes=3),
        TreeDepthDistributionRow(depth=2, num_nodes=1, cumulative_nodes=4),
    )


def test_dashboard_data_falls_back_to_newest_tree_export_when_metadata_is_stale(
    tmp_path: Path,
) -> None:
    """Dashboard tree data should fall back to the newest on-disk tree export."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.tree_snapshot_dir.mkdir(parents=True, exist_ok=True)
    save_training_tree_snapshot(
        TrainingTreeSnapshot(
            root_node_id="root-2",
            nodes=(
                TrainingNodeSnapshot(
                    node_id="root-2",
                    parent_ids=(),
                    child_ids=("leaf-2",),
                    depth=0,
                    state_ref_payload={"kind": "root-2"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=False,
                    is_exact=False,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
                TrainingNodeSnapshot(
                    node_id="leaf-2",
                    parent_ids=("root-2",),
                    child_ids=(),
                    depth=1,
                    state_ref_payload={"kind": "leaf-2"},
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    is_terminal=True,
                    is_exact=True,
                    over_event_label=None,
                    visit_count=1,
                    metadata={},
                ),
            ),
            metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
        ),
        paths.tree_snapshot_path_for_generation(2),
    )
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=0,
            latest_tree_snapshot_path="tree_exports/generation_000001.json",
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=1,
            last_save_unix_s=None,
        ),
        paths.run_state_path,
    )

    dashboard_data = build_morpion_bootstrap_dashboard_data(tmp_path)

    assert dashboard_data.run_summary.latest_tree_num_nodes == 2
    assert (
        dashboard_data.latest_tree_snapshot_status_message
        == "Tree snapshot metadata points to a missing file; using the newest tree export discovered on disk instead."
    )
    assert dashboard_data.latest_tree_depth_distribution == (
        TreeDepthDistributionRow(depth=0, num_nodes=1, cumulative_nodes=1),
        TreeDepthDistributionRow(depth=1, num_nodes=1, cumulative_nodes=2),
    )
