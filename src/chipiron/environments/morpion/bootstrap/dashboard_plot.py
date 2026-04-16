"""Matplotlib plotting helpers for Morpion bootstrap dashboard data."""

from __future__ import annotations

from datetime import datetime
from collections.abc import Iterable, Mapping, Sequence

from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from .history_view import (
    ActiveEvaluatorTimeSeriesPoint,
    IntTimeSeriesPoint,
    OptionalFloatTimeSeriesPoint,
    OptionalIntTimeSeriesPoint,
    TreeDepthDistributionRow,
)


def plot_tree_size(series: tuple[IntTimeSeriesPoint, ...]) -> None:
    """Plot tree size as a function of bootstrap timestamp."""
    axis = _new_axis()
    axis.set_title("Tree Size")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("num nodes")
    x_values, y_values = _int_points_with_timestamps(series)
    if not _plot_if_enough_points(axis, x_values, y_values, label="tree size"):
        return
    _format_datetime_axis(axis, x_values)


def plot_record_score(series: tuple[OptionalIntTimeSeriesPoint, ...]) -> None:
    """Plot canonical Morpion record score as moves since start."""
    axis = _new_axis()
    axis.set_title("Record Score")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("record (moves since start)")
    x_values, y_values = _non_none_optional_int_points(series)
    if not _plot_if_enough_points(
        axis,
        x_values,
        y_values,
        label="record (moves since start)",
    ):
        return
    _format_datetime_axis(axis, x_values)


def plot_certified_record_score(
    series: tuple[OptionalIntTimeSeriesPoint, ...],
) -> None:
    """Plot best-so-far certified Morpion record score against UTC time."""
    axis = _new_axis()
    axis.set_title("Certified Record Over Time")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("certified record (moves since start)")
    x_values, y_values = _non_none_optional_int_points(series)
    if not x_values or not y_values:
        _render_empty_message(axis, "No certified record yet.")
        return
    axis.plot(x_values, y_values, marker="o", label="certified record")
    _format_datetime_axis(axis, x_values, format_string="%Y-%m-%d %H:%M")


def plot_dataset_size(series: tuple[OptionalIntTimeSeriesPoint, ...]) -> None:
    """Plot dataset size as a function of bootstrap timestamp."""
    axis = _new_axis()
    axis.set_title("Dataset Size")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("rows")
    x_values, y_values = _non_none_optional_int_points(series)
    if not _plot_if_enough_points(axis, x_values, y_values, label="dataset rows"):
        return
    _format_datetime_axis(axis, x_values)


def plot_evaluator_losses(
    loss_by_name: Mapping[str, tuple[OptionalFloatTimeSeriesPoint, ...]],
) -> None:
    """Plot sparse evaluator-loss series, grouped by evaluator name."""
    axis = _new_axis()
    plotted_any_line = False
    plotted_x_values: list[datetime] = []
    for evaluator_name, series in sorted(loss_by_name.items()):
        x_values, y_values = _non_none_optional_float_points(series)
        if len(x_values) < 2:
            continue
        axis.plot(x_values, y_values, label=evaluator_name)
        plotted_any_line = True
        plotted_x_values.extend(x_values)

    axis.set_title("Evaluator Losses")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("loss")
    if not plotted_any_line:
        _render_waiting_for_history(axis)
        return
    axis.legend()
    _format_datetime_axis(axis, plotted_x_values)


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
        parsed_timestamp
        for point in series
        if (parsed_timestamp := _parse_timestamp_utc(point.timestamp_utc)) is not None
        if point.active_evaluator_name is not None
    ]
    axis.set_title("Active Evaluator")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("evaluator")
    if len(x_values) < 2:
        _render_waiting_for_history(axis)
        return

    y_values = [
        name_to_y[point.active_evaluator_name]
        for point in series
        if point.active_evaluator_name is not None
    ]
    axis.plot(x_values, y_values, label="active evaluator")
    axis.set_yticks(list(name_to_y.values()), list(name_to_y.keys()))
    _format_datetime_axis(axis, x_values)


def plot_tree_depth_distribution(
    rows: Sequence[TreeDepthDistributionRow],
) -> None:
    """Plot the latest saved tree node counts grouped by depth."""
    axis = _new_axis()
    axis.set_title("Current Tree Depth Distribution")
    axis.set_xlabel("depth")
    axis.set_ylabel("num nodes")
    if not rows:
        _render_empty_message(axis, "No tree depth distribution available yet.")
        return
    axis.bar(
        [row.depth for row in rows],
        [row.num_nodes for row in rows],
    )


def _new_axis() -> plt.Axes:
    """Create one fresh figure and return its default axis."""
    figure = plt.figure()
    return figure.gca()


def _non_none_optional_int_points(
    series: Sequence[OptionalIntTimeSeriesPoint],
) -> tuple[list[datetime], list[int]]:
    """Return timestamp/value lists for integer series points with known values."""
    x_values: list[datetime] = []
    y_values: list[int] = []
    for point in series:
        if point.value is None:
            continue
        timestamp = _parse_timestamp_utc(point.timestamp_utc)
        if timestamp is None:
            continue
        x_values.append(timestamp)
        y_values.append(point.value)
    return x_values, y_values


def _non_none_optional_float_points(
    series: Sequence[OptionalFloatTimeSeriesPoint],
) -> tuple[list[datetime], list[float]]:
    """Return timestamp/value lists for float series points with known values."""
    x_values: list[datetime] = []
    y_values: list[float] = []
    for point in series:
        if point.value is None:
            continue
        timestamp = _parse_timestamp_utc(point.timestamp_utc)
        if timestamp is None:
            continue
        x_values.append(timestamp)
        y_values.append(point.value)
    return x_values, y_values


def _int_points_with_timestamps(
    series: Sequence[IntTimeSeriesPoint],
) -> tuple[list[datetime], list[int]]:
    """Return timestamp/value lists for integer series points."""
    x_values: list[datetime] = []
    y_values: list[int] = []
    for point in series:
        timestamp = _parse_timestamp_utc(point.timestamp_utc)
        if timestamp is None:
            continue
        x_values.append(timestamp)
        y_values.append(point.value)
    return x_values, y_values


def _parse_timestamp_utc(timestamp_utc: str) -> datetime | None:
    """Parse one persisted UTC timestamp, accepting the ``Z`` suffix."""
    normalized = timestamp_utc.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _plot_if_enough_points(
    axis: plt.Axes,
    x_values: Sequence[datetime],
    y_values: Sequence[int | float],
    *,
    label: str,
) -> bool:
    """Plot one line only when enough points exist to form a time series."""
    if len(x_values) < 2 or len(y_values) < 2:
        _render_waiting_for_history(axis)
        return False
    axis.plot(x_values, y_values, label=label)
    return True


def _render_waiting_for_history(axis: plt.Axes) -> None:
    """Render a consistent empty-state message inside one plot axis."""
    _render_empty_message(axis, "Waiting for more history to draw this plot.")


def _render_empty_message(axis: plt.Axes, message: str) -> None:
    """Render one consistent empty-state message inside one plot axis."""
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _format_datetime_axis(
    axis: plt.Axes,
    x_values: Sequence[datetime],
    *,
    format_string: str | None = None,
) -> None:
    """Format one x-axis for readable UTC datetimes."""
    if not x_values:
        return
    span_seconds = (max(x_values) - min(x_values)).total_seconds()
    formatter = mdates.DateFormatter(
        (
            format_string
            if format_string is not None
            else "%H:%M:%S" if span_seconds < 3600 else "%m-%d %H:%M"
        ),
        tz=x_values[0].tzinfo,
    )
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.figure.autofmt_xdate(rotation=30, ha="right")


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
    "plot_certified_record_score",
    "plot_dataset_size",
    "plot_evaluator_losses",
    "plot_record_score",
    "plot_tree_depth_distribution",
    "plot_tree_size",
]
