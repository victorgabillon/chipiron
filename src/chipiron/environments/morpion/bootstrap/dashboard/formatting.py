"""Shared dashboard formatting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

MAX_PLOT_POINTS = 2000

__all__ = [
    "InvalidDashboardPlotPointLimitError",
    "control_float_value",
    "control_number_value",
    "downsample_loss_series_by_name",
    "downsample_series",
    "format_bool_icon",
    "format_disk_usage_pct",
    "format_seconds",
    "format_value",
    "latest_optional_value",
    "mapping_value",
    "numeric_value",
    "percentage",
    "ratio",
]


class InvalidDashboardPlotPointLimitError(ValueError):
    """Raised when dashboard plot downsampling is configured with an invalid cap."""

    def __init__(self, max_points: int) -> None:
        """Initialize the invalid plot-point-limit error."""
        super().__init__(
            f"Dashboard plot max_points must be at least 1, got {max_points}."
        )


def format_disk_usage_pct(value: float | None) -> str:
    """Format one optional disk-usage percentage for dashboard metrics."""
    if value is None:
        return "unknown"
    return f"{value:.2f}% of device"


def mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    """Return one nested string-keyed mapping from dashboard metadata."""
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        return {}
    raw_mapping = cast("Mapping[object, object]", value)
    if not all(isinstance(item_key, str) for item_key in raw_mapping):
        return {}
    return cast("Mapping[str, object]", raw_mapping)


def numeric_value(value: object) -> float | None:
    """Return one finite numeric value for derived dashboard metrics."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def ratio(
    numerator: object,
    denominator: object,
) -> float | None:
    """Return a safe ratio for optional numeric values."""
    normalized_numerator = numeric_value(numerator)
    normalized_denominator = numeric_value(denominator)
    if normalized_numerator is None or normalized_denominator is None:
        return None
    if normalized_denominator <= 0:
        return None
    return normalized_numerator / normalized_denominator


def percentage(value: object) -> str:
    """Format one optional ratio as a dashboard percentage."""
    normalized = numeric_value(value)
    if normalized is None:
        return "n/a"
    return f"{normalized * 100.0:.1f}%"


def downsample_series[SeriesPointT](
    series: Sequence[SeriesPointT],
    max_points: int = MAX_PLOT_POINTS,
) -> tuple[SeriesPointT, ...]:
    """Return one bounded series while preserving the first and last points."""
    if max_points < 1:
        raise InvalidDashboardPlotPointLimitError(max_points)
    if len(series) <= max_points:
        return tuple(series)
    if max_points == 1:
        return (series[-1],)
    last_index = len(series) - 1
    sampled_indices = tuple(
        int(sample_index * last_index / (max_points - 1))
        for sample_index in range(max_points)
    )
    return tuple(series[index] for index in sampled_indices)


def downsample_loss_series_by_name[SeriesPointT](
    loss_by_name: Mapping[str, Sequence[SeriesPointT]],
    max_points: int = MAX_PLOT_POINTS,
) -> dict[str, tuple[SeriesPointT, ...]]:
    """Return one bounded evaluator-loss mapping keyed by evaluator name."""
    return {
        evaluator_name: downsample_series(series, max_points=max_points)
        for evaluator_name, series in loss_by_name.items()
    }


def latest_optional_value(series: tuple[Any, ...]) -> object | None:
    """Return the latest value from one optional dashboard series."""
    if not series:
        return None
    return getattr(series[-1], "value", None)


def format_value(value: object | None) -> str:
    """Render optional values consistently in the dashboard."""
    return "n/a" if value is None else str(value)


def format_seconds(value: object | None) -> str:
    """Render optional duration values for compact metrics."""
    normalized = numeric_value(value)
    if normalized is None:
        return "n/a"
    return f"{normalized:.3f}s"


def format_bool_icon(value: bool | None) -> str:
    """Render one optional boolean with compact visual icons."""
    if value is True:
        return "✔"
    if value is False:
        return "✖"
    return "—"


def control_number_value(value: int | None, *, default: int = 0) -> int:
    """Return one Streamlit-safe integer input default."""
    return default if value is None else value


def control_float_value(value: float | None, *, default: float = 0.0) -> float:
    """Return one Streamlit-safe float input default."""
    return default if value is None else value
