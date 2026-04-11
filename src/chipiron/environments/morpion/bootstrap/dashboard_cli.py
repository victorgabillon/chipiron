"""Local CLI for inspecting Morpion bootstrap dashboard data."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from .dashboard_plot import (
    plot_active_evaluator,
    plot_dataset_size,
    plot_evaluator_losses,
    plot_record_score,
    plot_tree_size,
)
from .history_view import MorpionBootstrapDashboardData, build_morpion_bootstrap_dashboard_data


def run_dashboard_cli(work_dir: str | Path, *, plot: bool = True) -> None:
    """Print a local summary and optionally plot the main dashboard signals."""
    data = build_morpion_bootstrap_dashboard_data(work_dir)

    print(_render_run_summary(data))
    print()
    print(_render_evaluator_selection_summary(data))
    print()
    print(_render_record_progress_summary(data))

    if not plot:
        return

    plot_tree_size(data.tree_num_nodes)
    plot_record_score(data.canonical_record_score)
    plot_dataset_size(data.dataset_num_rows)
    plot_evaluator_losses(data.evaluator_loss_by_name)
    plot_active_evaluator(data.active_evaluator)
    plt.show()


def _render_run_summary(data: MorpionBootstrapDashboardData) -> str:
    """Render the top-level run summary for terminal output."""
    summary = data.run_summary
    latest_record = _format_latest_record(
        summary.latest_record_score,
        summary.latest_record_total_points,
    )
    return "\n".join(
        (
            "=== Morpion Bootstrap Run Summary ===",
            "cycles: "
            f"{summary.num_cycles} "
            f"(train: {summary.num_train_cycles}, no-train: {summary.num_no_train_cycles})",
            f"latest generation: {_format_optional_value(summary.latest_generation)}",
            f"latest tree size: {_format_optional_value(summary.latest_tree_num_nodes)}",
            f"latest record: {latest_record}",
            f"latest evaluator: {_format_optional_value(summary.latest_active_evaluator_name)}",
        )
    )


def _render_evaluator_selection_summary(
    data: MorpionBootstrapDashboardData,
) -> str:
    """Render evaluator selection summary for terminal output."""
    summary = data.evaluator_selection_summary
    usage_lines = ["usage:"]
    sorted_counts = sorted(
        summary.active_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if not sorted_counts:
        usage_lines.append("  none")
    else:
        usage_lines.extend(
            f"  {evaluator_name}: {count}"
            for evaluator_name, count in sorted_counts
        )

    return "\n".join(
        (
            "=== Evaluator Selection ===",
            f"latest: {_format_optional_value(summary.latest_active_evaluator_name)}",
            f"switches: {summary.num_switches}",
            *usage_lines,
        )
    )


def _render_record_progress_summary(data: MorpionBootstrapDashboardData) -> str:
    """Render canonical record progression summary for terminal output."""
    summary = data.record_progress_summary
    best_line = _format_best_record(summary.best_score, summary.first_cycle_reaching_best)
    return "\n".join(
        (
            "=== Record Progress ===",
            f"latest: {_format_optional_value(summary.latest_score)}",
            f"best: {best_line}",
        )
    )


def _format_latest_record(score: int | None, total_points: int | None) -> str:
    """Format the latest record using canonical score semantics."""
    if score is None:
        return "n/a"
    if total_points is None:
        return f"{score} moves"
    return f"{score} moves ({total_points} points)"


def _format_best_record(score: int | None, cycle_index: int | None) -> str:
    """Format the best-record summary line."""
    if score is None:
        return "n/a"
    if cycle_index is None:
        return str(score)
    return f"{score} (first reached at cycle {cycle_index})"


def _format_optional_value(value: object | None) -> str:
    """Render missing values consistently in CLI output."""
    return "n/a" if value is None else str(value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=str)
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    run_dashboard_cli(args.work_dir, plot=not args.no_plot)