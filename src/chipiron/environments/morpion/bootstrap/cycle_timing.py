"""Timing helpers shared by Morpion bootstrap cycle implementations."""

from __future__ import annotations

from datetime import UTC, datetime


def timestamp_utc_from_unix_s(timestamp_unix_s: float) -> str:
    """Format one Unix timestamp as an ISO 8601 UTC string."""
    timestamp = datetime.fromtimestamp(timestamp_unix_s, tz=UTC)
    timespec = "seconds" if timestamp.microsecond == 0 else "microseconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


def should_save_progress(
    *,
    current_tree_size: int,
    tree_size_at_last_save: int,
    now_unix_s: float,
    last_save_unix_s: float | None,
    save_after_tree_growth_factor: float,
    save_after_seconds: float,
) -> bool:
    """Return whether the bootstrap loop should checkpoint and retrain now."""
    if last_save_unix_s is None:
        return True
    if (
        tree_size_at_last_save > 0
        and current_tree_size >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return True
    return now_unix_s - last_save_unix_s >= save_after_seconds


def save_trigger_reason(
    *,
    current_tree_size: int,
    tree_size_at_last_save: int,
    now_unix_s: float,
    last_save_unix_s: float | None,
    save_after_tree_growth_factor: float,
    save_after_seconds: float,
) -> str | None:
    """Return the reason a save trigger fired, if any."""
    if last_save_unix_s is None:
        return "first_cycle"
    if (
        tree_size_at_last_save > 0
        and current_tree_size >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return "growth_factor_reached"
    if now_unix_s - last_save_unix_s >= save_after_seconds:
        return "time_elapsed"
    return None


__all__ = [
    "save_trigger_reason",
    "should_save_progress",
    "timestamp_utc_from_unix_s",
]
