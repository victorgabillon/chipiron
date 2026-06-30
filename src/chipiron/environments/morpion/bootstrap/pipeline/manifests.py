"""Manifest helpers for Morpion artifact-pipeline stages."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineDatasetStatus,
    MorpionPipelineDatasetStatusArtifact,
    MorpionPipelineGenerationManifest,
    MorpionPipelineTrainingStatus,
    load_pipeline_dataset_status_file,
    load_pipeline_manifest,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_status_file,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.record_status import (
        MorpionBootstrapFrontierStatus,
        MorpionBootstrapRecordStatus,
    )

LOGGER = logging.getLogger(__name__)


class MissingPipelineTreeSnapshotFileError(FileNotFoundError):
    """Raised when a pipeline dataset stage cannot find its tree snapshot."""

    @classmethod
    def from_path(
        cls, tree_snapshot_path: Path | None
    ) -> MissingPipelineTreeSnapshotFileError:
        """Build one missing-tree-snapshot error with the resolved path."""
        return cls(f"Pipeline tree snapshot does not exist: {tree_snapshot_path}")


class MissingPipelineRowsFileError(FileNotFoundError):
    """Raised when a pipeline training stage cannot find its rows file."""

    @classmethod
    def from_path(cls, rows_path: Path | None) -> MissingPipelineRowsFileError:
        """Build one missing-rows error with the resolved path."""
        return cls(f"Pipeline rows file does not exist: {rows_path}")


def manifest_tree_snapshot_required_error() -> ValueError:
    """Build the canonical missing tree snapshot path error."""
    return ValueError("manifest.tree_snapshot_path is required")


def manifest_rows_path_required_error() -> ValueError:
    """Build the canonical missing rows path error."""
    return ValueError("manifest.rows_path is required")


def pipeline_manifest_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the canonical manifest path for one pipeline generation."""
    return paths.pipeline_manifest_path_for_generation(generation)


def load_generation_manifest(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineGenerationManifest:
    """Load the persisted pipeline manifest for one generation."""
    return load_pipeline_manifest(pipeline_manifest_path(paths, generation))


def latest_prior_dataset_status_artifact(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineDatasetStatusArtifact | None:
    """Return the latest readable dataset-status artifact before one generation."""
    for previous_generation in range(generation - 1, -1, -1):
        status_path = paths.pipeline_dataset_status_path_for_generation(
            previous_generation
        )
        if not status_path.is_file():
            continue
        try:
            return load_pipeline_dataset_status_file(status_path)
        except (OSError, TypeError, ValueError):
            LOGGER.warning(
                "Skipping unreadable dataset status artifact: %s",
                status_path,
                exc_info=True,
            )
    return None


def resolve_previous_pipeline_record_status(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionBootstrapRecordStatus | None:
    """Return the previous record status for one pipeline dataset generation."""
    latest_dataset_status = latest_prior_dataset_status_artifact(
        paths=paths,
        generation=generation,
    )
    if latest_dataset_status is not None:
        return latest_dataset_status.record_status
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path).latest_record_status
    return None


def resolve_previous_pipeline_frontier_status(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionBootstrapFrontierStatus | None:
    """Return the previous frontier status for one pipeline dataset generation."""
    latest_dataset_status = latest_prior_dataset_status_artifact(
        paths=paths,
        generation=generation,
    )
    if latest_dataset_status is not None:
        return latest_dataset_status.frontier_status
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path).latest_frontier_status
    return None


def save_dataset_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    dataset_status: MorpionPipelineDatasetStatus,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated dataset-stage manifest and matching status file."""
    next_manifest = replace(manifest, dataset_status=dataset_status)
    save_pipeline_manifest(
        next_manifest, pipeline_manifest_path(paths, manifest.generation)
    )
    save_pipeline_dataset_status_file(
        generation=manifest.generation,
        dataset_status=next_manifest.dataset_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_dataset_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def save_training_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    training_status: MorpionPipelineTrainingStatus,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated training-stage manifest and matching status file."""
    next_manifest = replace(manifest, training_status=training_status)
    save_pipeline_manifest(
        next_manifest, pipeline_manifest_path(paths, manifest.generation)
    )
    save_pipeline_training_status_file(
        generation=manifest.generation,
        training_status=next_manifest.training_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_training_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def require_manifest_tree_snapshot_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest tree snapshot path for dataset extraction."""
    if manifest.tree_snapshot_path is None:
        raise manifest_tree_snapshot_required_error()
    return manifest.tree_snapshot_path


def require_manifest_rows_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest rows path for pipeline training."""
    if manifest.rows_path is None:
        raise manifest_rows_path_required_error()
    return manifest.rows_path
