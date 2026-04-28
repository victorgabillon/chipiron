"""Pipeline manifest persistence helpers for Morpion bootstrap workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionPipelineDatasetStatus,
    MorpionPipelineGenerationManifest,
    MorpionPipelineTrainingStatus,
    save_pipeline_active_model,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_status_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_paths import MorpionBootstrapPaths


def write_pipeline_status_files(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    timestamp_utc: str,
    dataset_status: MorpionPipelineDatasetStatus,
    training_status: MorpionPipelineTrainingStatus,
    metadata: Mapping[str, object],
) -> None:
    """Persist lightweight pipeline stage-status files for one generation."""
    save_pipeline_dataset_status_file(
        generation=generation,
        dataset_status=dataset_status,
        updated_at_utc=timestamp_utc,
        metadata=metadata,
        path=paths.pipeline_dataset_status_path_for_generation(generation),
    )
    save_pipeline_training_status_file(
        generation=generation,
        training_status=training_status,
        updated_at_utc=timestamp_utc,
        metadata=metadata,
        path=paths.pipeline_training_status_path_for_generation(generation),
    )


def write_pipeline_manifest_for_generation(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    timestamp_utc: str,
    relative_runtime_checkpoint_path: str | None,
    relative_tree_snapshot_path: str,
    relative_rows_path: str | None,
    model_bundle_paths: Mapping[str, str],
    selected_evaluator_name: str | None,
    dataset_status: MorpionPipelineDatasetStatus,
    training_status: MorpionPipelineTrainingStatus,
    metadata: Mapping[str, object],
) -> None:
    """Persist one pipeline manifest mirroring saved bootstrap artifacts."""
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc=timestamp_utc,
            runtime_checkpoint_path=relative_runtime_checkpoint_path,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=relative_rows_path,
            model_bundle_paths=dict(model_bundle_paths),
            selected_evaluator_name=selected_evaluator_name,
            dataset_status=dataset_status,
            training_status=training_status,
            metadata=dict(metadata),
        ),
        paths.pipeline_manifest_path_for_generation(generation),
    )
    write_pipeline_status_files(
        paths=paths,
        generation=generation,
        timestamp_utc=timestamp_utc,
        dataset_status=dataset_status,
        training_status=training_status,
        metadata=metadata,
    )


def write_pipeline_active_model(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    selected_evaluator_name: str,
    model_bundle_paths: Mapping[str, str],
    timestamp_utc: str,
) -> None:
    """Persist the currently selected active model for pipeline consumers."""
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=generation,
            evaluator_name=selected_evaluator_name,
            model_bundle_path=model_bundle_paths[selected_evaluator_name],
            updated_at_utc=timestamp_utc,
            metadata={"selection_policy": "lowest_final_loss"},
        ),
        paths.pipeline_active_model_path,
    )


__all__ = [
    "write_pipeline_active_model",
    "write_pipeline_manifest_for_generation",
    "write_pipeline_status_files",
]
