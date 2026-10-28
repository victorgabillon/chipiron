"""Manifest helpers for Morpion artifact-pipeline stages."""

import logging
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.dataset_publication import (
    enabled,
    project_dataset,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    manifest_read_pin,
    require,
    validate_manifest_publication,
)
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
    from collections.abc import Iterable, Iterator
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
    manifest = load_pipeline_manifest(pipeline_manifest_path(paths, generation))
    validate_manifest_publication(paths.work_dir, manifest)
    return manifest


@contextmanager
def pinned_generation_manifest(
    *, paths: MorpionBootstrapPaths, generation: int
) -> Iterator[MorpionPipelineGenerationManifest]:
    """Keep one selected generation pinned through its consuming operation."""
    path = pipeline_manifest_path(paths, generation)
    state = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.exists()
        else None
    )
    fenced = state is not None and enabled(state)
    manifest = load_pipeline_manifest(path) if path.exists() else None
    if fenced:
        assert state is not None
        manifest = project_dataset(paths, state, generation, manifest)
    require(manifest is not None, "Missing selected generation discovery manifest.")
    assert manifest is not None
    require(
        manifest.generation == generation, "Selected discovery generation mismatch."
    )
    with manifest_read_pin(paths.work_dir, manifest):
        if fenced:
            # Re-observe acceptance while holding the scientific lifetime pin.
            state = load_bootstrap_run_state(paths.run_state_path)
            manifest = project_dataset(
                paths, state, generation, manifest, verify_rows=True
            )
        from chipiron.environments.morpion.bootstrap.training_publication import (
            enabled as training_enabled,
        )
        from chipiron.environments.morpion.bootstrap.training_publication import (
            project_training,
        )

        if state is not None and training_enabled(state):
            manifest = project_training(paths, state, manifest, verify=True)
        yield manifest


def verified_generation_rows[T](
    rows: Iterable[T],
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
) -> Iterator[T]:
    """Recheck source bytes before the existing streaming writer replaces its output."""
    yield from rows
    validate_manifest_publication(paths.work_dir, manifest)


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
        except OSError, TypeError, ValueError:
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
