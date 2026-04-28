"""Phase 3 artifact-pipeline stage entrypoints for Morpion bootstrap."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows,
)

from .bootstrap_loop import (
    MorpionBootstrapArgs,
    MorpionSearchRunner,
    _extract_rows_from_training_snapshot,
    _require_artifact_pipeline_mode,
    _run_bootstrap_loop_impl,
    _timestamp_utc_from_unix_s,
    _train_and_select_evaluators,
)
from .bootstrap_memory import log_after_cycle_gc, memory_diagnostics_config_from_args
from .bootstrap_paths import MorpionBootstrapPaths
from .control import load_bootstrap_control
from .memory_diagnostics import MemoryDiagnostics
from .pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
    save_pipeline_active_model,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_status_file,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _manifest_tree_snapshot_required_error() -> ValueError:
    """Build the canonical missing tree snapshot path error."""
    return ValueError("manifest.tree_snapshot_path is required")


def _manifest_rows_path_required_error() -> ValueError:
    """Build the canonical missing rows path error."""
    return ValueError("manifest.rows_path is required")


def _dataset_stage_requires_done_status_error() -> ValueError:
    """Build the canonical dataset-ready error for training stage entry."""
    return ValueError("manifest.dataset_status == 'done' is required")


def _raise_missing_tree_snapshot_file_error(tree_snapshot_path: Path | None) -> NoReturn:
    """Raise the canonical dataset-stage missing snapshot file error."""
    del tree_snapshot_path
    raise FileNotFoundError


def _raise_missing_rows_file_error(rows_path: Path | None) -> NoReturn:
    """Raise the canonical training-stage missing rows file error."""
    del rows_path
    raise FileNotFoundError


def _pipeline_manifest_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the canonical manifest path for one pipeline generation."""
    return paths.pipeline_manifest_path_for_generation(generation)


def _now_timestamp_utc() -> str:
    """Return the current UTC timestamp formatted like the bootstrap loop."""
    return _timestamp_utc_from_unix_s(time.time())


def _load_generation_manifest(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineGenerationManifest:
    """Load the persisted pipeline manifest for one generation."""
    return load_pipeline_manifest(_pipeline_manifest_path(paths, generation))


def _save_dataset_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    dataset_status: str,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated dataset-stage manifest and matching status file."""
    next_manifest = replace(manifest, dataset_status=dataset_status)
    save_pipeline_manifest(next_manifest, _pipeline_manifest_path(paths, manifest.generation))
    save_pipeline_dataset_status_file(
        generation=manifest.generation,
        dataset_status=next_manifest.dataset_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_dataset_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def _save_training_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    training_status: str,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated training-stage manifest and matching status file."""
    next_manifest = replace(manifest, training_status=training_status)
    save_pipeline_manifest(next_manifest, _pipeline_manifest_path(paths, manifest.generation))
    save_pipeline_training_status_file(
        generation=manifest.generation,
        training_status=next_manifest.training_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_training_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def _require_manifest_tree_snapshot_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest tree snapshot path for dataset extraction."""
    if manifest.tree_snapshot_path is None:
        raise _manifest_tree_snapshot_required_error()
    return manifest.tree_snapshot_path


def _require_manifest_rows_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest rows path for pipeline training."""
    if manifest.rows_path is None:
        raise _manifest_rows_path_required_error()
    return manifest.rows_path


def run_pipeline_growth_stage(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_cycles: int = 1,
) -> MorpionBootstrapRunState:
    """Run the conservative Phase 3 growth stage for artifact-pipeline mode."""
    _require_artifact_pipeline_mode(args)
    LOGGER.info("[pipeline] growth_start max_cycles=%s", max_cycles)
    # TODO: Split growth/export/checkpoint from train/select once Phase 4 wires a real orchestrator.
    run_state = _run_bootstrap_loop_impl(args, runner, max_cycles=max_cycles)
    LOGGER.info(
        "[pipeline] growth_done generation=%s cycle=%s",
        run_state.generation,
        run_state.cycle_index,
    )
    return run_state


def run_pipeline_dataset_stage(
    args: MorpionBootstrapArgs,
    *,
    generation: int,
) -> MorpionPipelineGenerationManifest:
    """Extract supervised rows for one persisted pipeline generation."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    manifest = _load_generation_manifest(paths=paths, generation=generation)
    timestamp_utc = _now_timestamp_utc()
    LOGGER.info("[pipeline] dataset_start generation=%s", generation)
    manifest = _save_dataset_manifest_status(
        paths=paths,
        manifest=manifest,
        dataset_status="extracting_rows",
        timestamp_utc=timestamp_utc,
    )
    try:
        tree_snapshot_path = paths.resolve_work_dir_path(
            _require_manifest_tree_snapshot_path(manifest)
        )
        if tree_snapshot_path is None or not tree_snapshot_path.is_file():
            _raise_missing_tree_snapshot_file_error(tree_snapshot_path)
        from anemone.training_export import load_training_tree_snapshot

        snapshot = load_training_tree_snapshot(tree_snapshot_path)
        rows = _extract_rows_from_training_snapshot(
            args=args,
            snapshot=snapshot,
            generation=generation,
        )
        rows_path = (
            paths.resolve_work_dir_path(manifest.rows_path)
            if manifest.rows_path is not None
            else paths.rows_path_for_generation(generation)
        )
        if rows_path is None:
            rows_path = paths.rows_path_for_generation(generation)
        save_morpion_supervised_rows(rows, rows_path)
        timestamp_utc = _now_timestamp_utc()
        manifest = replace(
            manifest,
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
        )
        save_pipeline_manifest(manifest, _pipeline_manifest_path(paths, generation))
        save_pipeline_dataset_status_file(
            generation=generation,
            dataset_status=manifest.dataset_status,
            updated_at_utc=timestamp_utc,
            metadata=manifest.metadata,
            path=paths.pipeline_dataset_status_path_for_generation(generation),
        )
        LOGGER.info(
            "[pipeline] dataset_done generation=%s rows=%s",
            generation,
            len(rows.rows),
        )
    except Exception:
        timestamp_utc = _now_timestamp_utc()
        _save_dataset_manifest_status(
            paths=paths,
            manifest=manifest,
            dataset_status="failed",
            timestamp_utc=timestamp_utc,
        )
        LOGGER.exception("[pipeline] dataset_fail generation=%s", generation)
        raise
    else:
        return manifest


def run_pipeline_training_stage(
    args: MorpionBootstrapArgs,
    *,
    generation: int,
) -> MorpionPipelineGenerationManifest:
    """Train evaluators and select the active model for one pipeline generation."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    manifest = _load_generation_manifest(paths=paths, generation=generation)
    if manifest.dataset_status != "done":
        raise _dataset_stage_requires_done_status_error()
    timestamp_utc = _now_timestamp_utc()
    LOGGER.info("[pipeline] training_start generation=%s", generation)
    manifest = _save_training_manifest_status(
        paths=paths,
        manifest=manifest,
        training_status="training",
        timestamp_utc=timestamp_utc,
    )
    try:
        rows_path = paths.resolve_work_dir_path(_require_manifest_rows_path(manifest))
        if rows_path is None or not rows_path.is_file():
            _raise_missing_rows_file_error(rows_path)
        rows = load_morpion_supervised_rows(rows_path)
        run_state = (
            load_bootstrap_run_state(paths.run_state_path)
            if paths.run_state_path.is_file()
            else initialize_bootstrap_run_state()
        )
        resolved_control = load_bootstrap_control(paths.control_path)
        memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
        try:
            training_result = _train_and_select_evaluators(
                args=args,
                paths=paths,
                run_state=run_state,
                rows=rows,
                rows_path=rows_path,
                generation=generation,
                timestamp_utc=timestamp_utc,
                resolved_evaluators_config=args.resolved_evaluators_config(),
                resolved_control=resolved_control,
                memory=memory,
            )
        finally:
            log_after_cycle_gc(memory)
            memory.close()
        timestamp_utc = _now_timestamp_utc()
        manifest = replace(
            manifest,
            model_bundle_paths=training_result.model_bundle_paths,
            selected_evaluator_name=training_result.selected_evaluator_name,
            training_status="done",
        )
        save_pipeline_manifest(manifest, _pipeline_manifest_path(paths, generation))
        save_pipeline_training_status_file(
            generation=generation,
            training_status=manifest.training_status,
            updated_at_utc=timestamp_utc,
            metadata=manifest.metadata,
            path=paths.pipeline_training_status_path_for_generation(generation),
        )
        save_pipeline_active_model(
            MorpionPipelineActiveModel(
                generation=generation,
                evaluator_name=training_result.selected_evaluator_name,
                model_bundle_path=training_result.model_bundle_paths[
                    training_result.selected_evaluator_name
                ],
                updated_at_utc=timestamp_utc,
                metadata={"selection_policy": "lowest_final_loss"},
            ),
            paths.pipeline_active_model_path,
        )
        LOGGER.info(
            "[pipeline] training_done generation=%s selected=%s",
            generation,
            training_result.selected_evaluator_name,
        )
    except Exception:
        timestamp_utc = _now_timestamp_utc()
        _save_training_manifest_status(
            paths=paths,
            manifest=manifest,
            training_status="failed",
            timestamp_utc=timestamp_utc,
        )
        LOGGER.exception("[pipeline] training_fail generation=%s", generation)
        raise
    else:
        return manifest


__all__ = [
    "run_pipeline_dataset_stage",
    "run_pipeline_growth_stage",
    "run_pipeline_training_stage",
]
