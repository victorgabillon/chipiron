"""Tests for the local Morpion bootstrap dashboard CLI and plots."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

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

from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapPaths,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
    plot_active_evaluator,
    plot_dataset_size,
    plot_evaluator_losses,
    plot_record_score,
    plot_tree_size,
    run_dashboard_cli,
)

if TYPE_CHECKING:
    import pytest


def _make_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    tree_num_nodes: int,
    record_score: int | None,
    total_points: int | None,
    active_evaluator_name: str | None,
    dataset_num_rows: int | None = None,
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] | None = None,
) -> MorpionBootstrapEvent:
    """Build one representative bootstrap event for dashboard tests."""
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
        training=MorpionBootstrapTrainingStatus(triggered=True),
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


def test_dashboard_cli_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The dashboard CLI should print a readable summary without plotting."""
    recorder = MorpionBootstrapHistoryRecorder(
        MorpionBootstrapPaths.from_work_dir(tmp_path).history_paths()
    )
    recorder.record(
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
            tree_num_nodes=10,
            record_score=12,
            total_points=48,
            active_evaluator_name="linear",
            dataset_num_rows=10,
        )
    )

    run_dashboard_cli(tmp_path, plot=False)

    captured = capsys.readouterr().out
    assert "=== Morpion Bootstrap Run Summary ===" in captured
    assert "cycles: 1 (train: 1, no-train: 0)" in captured
    assert "latest record: 12 moves (48 points)" in captured
    assert "=== Evaluator Selection ===" in captured
    assert "linear: 1" in captured
    assert "=== Record Progress ===" in captured
    assert "best: 12 (first reached at cycle 0)" in captured


def test_plot_functions_accept_empty_series() -> None:
    """All dashboard plot helpers should tolerate empty inputs."""
    plt.close("all")

    plot_tree_size(())
    plot_record_score(())
    plot_dataset_size(())
    plot_evaluator_losses({})
    plot_active_evaluator(())

    assert len(plt.get_fignums()) == 5
    plt.close("all")


def test_dashboard_cli_runs_with_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dashboard CLI should exercise all plot helpers and show once."""
    recorder = MorpionBootstrapHistoryRecorder(
        MorpionBootstrapPaths.from_work_dir(tmp_path).history_paths()
    )
    recorder.record(
        _make_event(
            cycle_index=0,
            generation=1,
            timestamp_utc="2026-04-11T08:00:00Z",
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
    )

    show_calls: list[bool] = []

    def _fake_show() -> None:
        show_calls.append(True)

    plt.close("all")
    monkeypatch.setattr(plt, "show", _fake_show)

    run_dashboard_cli(tmp_path, plot=True)

    assert show_calls == [True]
    assert len(plt.get_fignums()) == 5
    plt.close("all")
