"""Matplotlib plotting helpers for Morpion bootstrap dashboard data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from matplotlib import pyplot as plt

from .history_view import (
    ActiveEvaluatorTimeSeriesPoint,
    IntTimeSeriesPoint,
    OptionalFloatTimeSeriesPoint,
    OptionalIntTimeSeriesPoint,
)


def plot_tree_size(series: tuple[IntTimeSeriesPoint, ...]) -> None:
    """Plot tree size as a function of bootstrap cycle."""
    axis = _new_axis()
    axis.plot(
        [point.cycle_index for point in series],
        [point.value for point in series],
        label="tree size",
    )
    axis.set_title("Tree Size")
    axis.set_xlabel("cycle")
    axis.set_ylabel("num nodes")


def plot_record_score(series: tuple[OptionalIntTimeSeriesPoint, ...]) -> None:
    """Plot canonical Morpion record score as moves since start."""
    axis = _new_axis()
    x_values, y_values = _non_none_optional_int_points(series)
    axis.plot(x_values, y_values, label="record (moves since start)")
    axis.set_title("Record Score")
    axis.set_xlabel("cycle")
    axis.set_ylabel("record (moves since start)")


def plot_dataset_size(series: tuple[OptionalIntTimeSeriesPoint, ...]) -> None:
    """Plot dataset size as a function of bootstrap cycle."""
    axis = _new_axis()
    x_values, y_values = _non_none_optional_int_points(series)
    axis.plot(x_values, y_values, label="dataset rows")
    axis.set_title("Dataset Size")
    axis.set_xlabel("cycle")
    axis.set_ylabel("rows")


def plot_evaluator_losses(
    loss_by_name: Mapping[str, tuple[OptionalFloatTimeSeriesPoint, ...]],
) -> None:
    """Plot sparse evaluator-loss series, grouped by evaluator name."""
    axis = _new_axis()
    plotted_any_line = False
    for evaluator_name, series in sorted(loss_by_name.items()):
        x_values, y_values = _non_none_optional_float_points(series)
        if not x_values:
            continue
        axis.plot(x_values, y_values, label=evaluator_name)
        plotted_any_line = True

    axis.set_title("Evaluator Losses")
    axis.set_xlabel("cycle")
    axis.set_ylabel("loss")
    if plotted_any_line:
        axis.legend()


def plot_active_evaluator(
    series: tuple[ActiveEvaluatorTimeSeriesPoint, ...],
) -> None:
    """Plot the active evaluator timeline using integer y-axis buckets."""
    axis = _new_axis()
    evaluator_names = _ordered_non_none_names(
        point.active_evaluator_name for point in series
    )
    name_to_y = {
        evaluator_name: index
        for index, evaluator_name in enumerate(evaluator_names)
    }
    x_values = [
        point.cycle_index
        for point in series
        if point.active_evaluator_name is not None
    ]
    y_values = [
        name_to_y[point.active_evaluator_name]
        for point in series
        if point.active_evaluator_name is not None
    ]
    axis.plot(x_values, y_values, label="active evaluator")
    axis.set_title("Active Evaluator")
    axis.set_xlabel("cycle")
    axis.set_ylabel("evaluator")
    axis.set_yticks(list(name_to_y.values()), list(name_to_y.keys()))


def _new_axis() -> plt.Axes:
    """Create one fresh figure and return its default axis."""
    figure = plt.figure()
    return figure.gca()


def _non_none_optional_int_points(
    series: Sequence[OptionalIntTimeSeriesPoint],
) -> tuple[list[int], list[int]]:
    """Return cycle/value lists for integer series points with known values."""
    x_values: list[int] = []
    y_values: list[int] = []
    for point in series:
        if point.value is None:
            continue
        x_values.append(point.cycle_index)
        y_values.append(point.value)
    return x_values, y_values


def _non_none_optional_float_points(
    series: Sequence[OptionalFloatTimeSeriesPoint],
) -> tuple[list[int], list[float]]:
    """Return cycle/value lists for float series points with known values."""
    x_values: list[int] = []
    y_values: list[float] = []
    for point in series:
        if point.value is None:
            continue
        x_values.append(point.cycle_index)
        y_values.append(point.value)
    return x_values, y_values


def _ordered_non_none_names(names: Iterable[str | None]) -> list[str]:
    """Return evaluator names in first-seen order without duplicates."""
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    for name in names:
        if name is None or name in seen_names:
            continue
        ordered_names.append(name)
        seen_names.add(name)
    return ordered_names


__all__ = [
    "plot_active_evaluator",
    "plot_dataset_size",
    "plot_evaluator_losses",
    "plot_record_score",
    "plot_tree_size",
]