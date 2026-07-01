"""Cached dashboard data loading and freshness tokens."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
    build_current_certified_record_board_view,
    build_morpion_bootstrap_dashboard_data,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        MorpionBootstrapCertifiedRecordBoardView,
    )

__all__ = [
    "cached_build_current_certified_record_board_view",
    "cached_build_morpion_bootstrap_dashboard_data",
    "cached_certified_record_board_freshness_tokens",
    "cached_dashboard_data_freshness_tokens",
]


def _path_mtime_ns(path: Path) -> int:
    """Return one path freshness token, treating missing paths as zero."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _loss_series_contains_points(
    loss_by_name: Mapping[str, tuple[Any, ...]],
) -> bool:
    """Return whether any evaluator loss series contains at least one concrete point."""
    return any(series for series in loss_by_name.values())


def _checked_training_status_files_summary(paths: MorpionBootstrapPaths) -> str:
    """Summarize the training-status files the dashboard inspected for loss data."""
    checked_files = sorted(paths.pipeline_dir.glob("generation_*/training_status.json"))
    if not checked_files:
        return "none"
    rendered = [paths.relative_to_work_dir(path) for path in checked_files[-5:]]
    omitted_count = len(checked_files) - len(rendered)
    if omitted_count > 0:
        return ", ".join(rendered) + f" (+{omitted_count} more)"
    return ", ".join(rendered)


def _latest_generation_json_path(directory: Path) -> Path | None:
    """Return the newest generation JSON file in one directory when present."""
    candidates = sorted(directory.glob("generation_*.json"))
    return None if not candidates else candidates[-1]


def _latest_tree_snapshot_generation_json_path(
    paths: MorpionBootstrapPaths,
) -> Path | None:
    """Return the newest flat or sharded tree generation JSON file."""
    candidates = [
        path
        for path in (
            _latest_generation_json_path(paths.tree_snapshot_dir),
            _latest_generation_json_path(paths.sharded_tree_snapshot_dir),
        )
        if path is not None
    ]
    return (
        None if not candidates else sorted(candidates, key=lambda path: path.name)[-1]
    )


def cached_dashboard_data_freshness_tokens(
    paths: MorpionBootstrapPaths,
) -> tuple[int, ...]:
    """Return freshness tokens for dashboard-wide data rebuilds."""
    latest_tree_snapshot_path = _latest_tree_snapshot_generation_json_path(paths)
    latest_runtime_checkpoint_path = _latest_generation_json_path(
        paths.runtime_checkpoint_dir
    )
    return (
        _path_mtime_ns(paths.work_dir),
        _path_mtime_ns(paths.bootstrap_config_path),
        _path_mtime_ns(paths.control_path),
        _path_mtime_ns(paths.run_state_path),
        _path_mtime_ns(paths.history_jsonl_path),
        _path_mtime_ns(paths.latest_status_path),
        _path_mtime_ns(paths.launcher_pid_path),
        _path_mtime_ns(paths.launcher_process_state_path),
        _path_mtime_ns(paths.launcher_stdout_log_path),
        _path_mtime_ns(paths.launcher_stderr_log_path),
        _path_mtime_ns(paths.tree_snapshot_dir),
        _path_mtime_ns(paths.sharded_tree_snapshot_dir),
        _path_mtime_ns(paths.runtime_checkpoint_dir),
        _path_mtime_ns(paths.rows_dir),
        _path_mtime_ns(paths.model_dir),
        _path_mtime_ns(paths.latest_linoo_selection_table_path),
        0
        if latest_tree_snapshot_path is None
        else _path_mtime_ns(latest_tree_snapshot_path),
        (
            0
            if latest_runtime_checkpoint_path is None
            else _path_mtime_ns(latest_runtime_checkpoint_path)
        ),
    )


def cached_certified_record_board_freshness_tokens(
    paths: MorpionBootstrapPaths,
) -> tuple[int, ...]:
    """Return freshness tokens for certified-record board rebuilds."""
    latest_tree_snapshot_path = _latest_tree_snapshot_generation_json_path(paths)
    return (
        _path_mtime_ns(paths.run_state_path),
        _path_mtime_ns(paths.history_jsonl_path),
        _path_mtime_ns(paths.tree_snapshot_dir),
        _path_mtime_ns(paths.sharded_tree_snapshot_dir),
        0
        if latest_tree_snapshot_path is None
        else _path_mtime_ns(latest_tree_snapshot_path),
    )


@lru_cache(maxsize=1)
def cached_build_morpion_bootstrap_dashboard_data(
    work_dir: str,
    freshness_tokens: tuple[int, ...],
) -> Any:
    """Cache dashboard-wide data for one work dir until relevant artifacts change."""
    _ = freshness_tokens
    return build_morpion_bootstrap_dashboard_data(work_dir)


@lru_cache(maxsize=1)
def cached_build_current_certified_record_board_view(
    work_dir: str,
    freshness_tokens: tuple[int, ...],
) -> MorpionBootstrapCertifiedRecordBoardView | None:
    """Cache the certified-record board view until its source artifacts change."""
    _ = freshness_tokens
    return build_current_certified_record_board_view(work_dir)
