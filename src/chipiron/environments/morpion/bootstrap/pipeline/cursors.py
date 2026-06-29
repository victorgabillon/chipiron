"""Training-cursor helpers for Morpion artifact-pipeline stages."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineTrainingCursor,
    load_pipeline_active_model,
    load_pipeline_training_cursor,
    save_pipeline_training_cursor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )


def _optional_generation_max(left: int | None, right: int) -> int:
    """Return max for an optional generation value and a concrete generation."""
    return max(left if left is not None else -1, right)


def _active_model_generation_for_training_guard(
    paths: MorpionBootstrapPaths,
) -> int | None:
    """Return current active-model generation when the singleton artifact exists."""
    if not paths.pipeline_active_model_path.is_file():
        return None
    return load_pipeline_active_model(paths.pipeline_active_model_path).generation


def _training_lower_bound_generation(paths: MorpionBootstrapPaths) -> int:
    """Return the monotonic lower bound for an explicit training stage."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    active_generation = _active_model_generation_for_training_guard(paths)
    return max(
        active_generation if active_generation is not None else -1,
        (
            cursor.latest_started_generation
            if cursor.latest_started_generation is not None
            else -1
        ),
        (
            cursor.latest_completed_generation
            if cursor.latest_completed_generation is not None
            else -1
        ),
    )


def _save_training_cursor_started(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has started training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_started_generation=_optional_generation_max(
            cursor.latest_started_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor


def _training_rows_subset_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the debug training-row subset path for one generation."""
    return paths.rows_dir / f"generation_{generation:06d}.training_subset.json"


def _save_training_cursor_completed(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has completed training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_completed_generation=_optional_generation_max(
            cursor.latest_completed_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor
