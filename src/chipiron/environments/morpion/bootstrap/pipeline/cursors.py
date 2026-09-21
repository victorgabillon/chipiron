"""Training-cursor helpers for Morpion artifact-pipeline stages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
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

LOGGER = logging.getLogger(__name__)

_GENERATION_DIR_RE = re.compile(r"^generation_(\d{6})$")


@dataclass(frozen=True, slots=True)
class MorpionTrainingLowerBound:
    """Resolved local lower bound for Morpion pipeline training selection."""

    active_model_generation: int | None
    active_model_source_generation: int | None
    active_model_source: str
    cursor_started_generation: int | None
    cursor_completed_generation: int | None
    generation: int


def optional_generation_max(left: int | None, right: int) -> int:
    """Return max for an optional generation value and a concrete generation."""
    return max(left if left is not None else -1, right)


def active_model_generation_for_training_guard(
    paths: MorpionBootstrapPaths,
) -> int | None:
    """Return current active-model generation when the singleton artifact exists."""
    if not paths.pipeline_active_model_path.is_file():
        return None
    return load_pipeline_active_model(paths.pipeline_active_model_path).generation


def _latest_local_generation(paths: MorpionBootstrapPaths) -> int | None:
    """Return the latest generation manifest produced in the current work dir."""
    if not paths.pipeline_dir.is_dir():
        return None
    latest_generation: int | None = None
    for child in paths.pipeline_dir.iterdir():
        if not child.is_dir():
            continue
        match = _GENERATION_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        generation = int(match.group(1))
        latest_generation = (
            generation
            if latest_generation is None
            else max(latest_generation, generation)
        )
    return latest_generation


def _active_model_has_local_generation_artifact(
    *,
    paths: MorpionBootstrapPaths,
    active_model: MorpionPipelineActiveModel,
) -> bool:
    """Return whether the active model generation exists in local pipeline state."""
    return (
        paths.pipeline_manifest_path_for_generation(active_model.generation).is_file()
        or paths.pipeline_training_status_path_for_generation(
            active_model.generation
        ).is_file()
    )


def _active_model_source_for_local_training_bound(
    *,
    paths: MorpionBootstrapPaths,
    active_model: MorpionPipelineActiveModel | None,
) -> str:
    """Classify active model provenance for local training lower-bound purposes."""
    if active_model is None:
        return "none"
    if active_model.source == "external_seed":
        return "external_seed"
    if not active_model.source_was_inferred:
        return "local_training"
    latest_local_generation = _latest_local_generation(paths)
    if (
        latest_local_generation is not None
        and active_model.generation > latest_local_generation
        and not _active_model_has_local_generation_artifact(
            paths=paths,
            active_model=active_model,
        )
    ):
        return "external_seed"
    return "local_training"


def training_lower_bound_details(
    paths: MorpionBootstrapPaths,
) -> MorpionTrainingLowerBound:
    """Return local training lower-bound details with active-model provenance split."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    active_model = (
        load_pipeline_active_model(paths.pipeline_active_model_path)
        if paths.pipeline_active_model_path.is_file()
        else None
    )
    active_model_source = _active_model_source_for_local_training_bound(
        paths=paths,
        active_model=active_model,
    )
    cursor_lower_bound = max(
        cursor.latest_started_generation
        if cursor.latest_started_generation is not None
        else 0,
        cursor.latest_completed_generation
        if cursor.latest_completed_generation is not None
        else 0,
    )
    active_local_generation = (
        None
        if active_model is None or active_model_source == "external_seed"
        else active_model.local_trained_generation
    )
    lower_bound = max(
        cursor_lower_bound,
        active_local_generation if active_local_generation is not None else 0,
    )
    details = MorpionTrainingLowerBound(
        active_model_generation=None
        if active_model is None
        else active_model.generation,
        active_model_source_generation=(
            None if active_model is None else active_model.source_generation
        ),
        active_model_source=active_model_source,
        cursor_started_generation=cursor.latest_started_generation,
        cursor_completed_generation=cursor.latest_completed_generation,
        generation=lower_bound,
    )
    LOGGER.info(
        "[training] active_model_source_generation=%s local_training_lower_bound=%s source=%s cursor_started=%s cursor_completed=%s active_model_generation=%s",
        "none"
        if details.active_model_source_generation is None
        else details.active_model_source_generation,
        details.generation,
        details.active_model_source,
        "none"
        if details.cursor_started_generation is None
        else details.cursor_started_generation,
        "none"
        if details.cursor_completed_generation is None
        else details.cursor_completed_generation,
        "none"
        if details.active_model_generation is None
        else details.active_model_generation,
    )
    return details


def training_lower_bound_generation(paths: MorpionBootstrapPaths) -> int:
    """Return the monotonic lower bound for an explicit training stage."""
    return training_lower_bound_details(paths).generation


def save_training_cursor_started(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has started training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_started_generation=optional_generation_max(
            cursor.latest_started_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor


def training_rows_subset_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the debug training-row subset path for one generation."""
    return paths.rows_dir / f"generation_{generation:06d}.training_subset.json"


def save_training_cursor_completed(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has completed training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_completed_generation=optional_generation_max(
            cursor.latest_completed_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor
