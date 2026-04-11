"""Tests for dashboard-ready Morpion bootstrap history views."""
# ruff: noqa: E402

from __future__ import annotations

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
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionBootstrapRunView,
    MorpionEvaluatorMetrics,
    MorpionRecordProgressSummary,
    active_evaluator_series,
    build_morpion_bootstrap_dashboard_data,
    canonical_record_score_series,
    dataset_num_rows_series,
    evaluator_loss_series_by_name,
    load_morpion_bootstrap_run_view,
    record_total_points_series,
    save_bootstrap_run_state,
    summarize_bootstrap_run,
    summarize_evaluator_selection,
    summarize_record_progress,
    training_triggered_series,
    tree_num_nodes_series,
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
        tree=MorpionBootstrapTreeStatus(num_nodes=tree_num_nodes),
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
            current_best_source="snapshot_exact_node" if record_score is not None else None,
        ),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=None,
            rows_path=None,
        ),
        evaluators={} if evaluator_metrics is None else dict(evaluator_metrics),
        metadata=metadata,
    )


def _make_run_state() -> MorpionBootstrapRunState:
    """Build one representative run state for run-view loading tests."""
    return MorpionBootstrapRunState(
        generation=2,
        cycle_index=3,
        latest_tree_snapshot_path="tree_exports/generation_000002.json",
        latest_rows_path="rows/generation_000002.json",
        latest_model_bundle_paths={"mlp": "models/generation_000002/mlp"},
        active_evaluator_name="mlp",
        tree_size_at_last_save=25,
        last_save_unix_s=200.0,
        latest_record_status=MorpionBootstrapRecordStatus(
            variant="5T",
            initial_pattern="greek_cross",
            initial_point_count=36,
            current_best_moves_since_start=19,
            current_best_total_points=55,
            current_best_is_exact=True,
            current_best_source="snapshot_exact_node",
        ),
    )


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

    assert run_view == MorpionBootstrapRunView(
        work_dir=paths.work_dir,
        run_state=run_state,
        latest_status=run_view.latest_status,
        history=(first_event, second_event),
    )
    assert run_view.latest_status.latest_event == second_event
    assert run_view.latest_status.latest_cycle_index == 1
    assert run_view.latest_status.latest_generation == 2


def test_empty_run_view_works(tmp_path: Path) -> None:
    """An empty work directory should still load as a valid run view."""
    run_view = load_morpion_bootstrap_run_view(tmp_path)

    assert run_view.work_dir == tmp_path.resolve()
    assert run_view.run_state is None
    assert run_view.history == ()
    assert run_view.latest_status.latest_event is None
    assert run_view.latest_status.latest_cycle_index is None
    assert run_view.latest_status.latest_generation is None


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

    dashboard_data = build_morpion_bootstrap_dashboard_data(tmp_path)

    assert isinstance(dashboard_data, MorpionBootstrapDashboardData)
    assert dashboard_data.run_summary.num_cycles == 2
    assert dashboard_data.run_summary.latest_record_score == 14
    assert dashboard_data.evaluator_selection_summary.latest_active_evaluator_name == "mlp"
    assert dashboard_data.record_progress_summary.best_score == 14
    assert len(dashboard_data.tree_num_nodes) == 2
    assert len(dashboard_data.canonical_record_score) == 2
    assert len(dashboard_data.record_total_points) == 2
    assert len(dashboard_data.dataset_num_rows) == 2
    assert set(dashboard_data.evaluator_loss_by_name) == {"linear"}
    assert len(dashboard_data.active_evaluator) == 2
    assert training_triggered_series((first_event, second_event)) == (
        (0, True),
        (1, False),
    )
    assert tuple(point.value for point in tree_num_nodes_series((first_event, second_event))) == (
        10,
        15,
    )
    assert tuple(point.value for point in dataset_num_rows_series((first_event, second_event))) == (
        10,
        None,
    )
