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

from .bootstrap_errors import MissingSavedBootstrapArtifactError
from .bootstrap_memory import log_after_cycle_gc, memory_diagnostics_config_from_args
from .bootstrap_paths import MorpionBootstrapPaths
from .config import (
    MorpionBootstrapConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    load_bootstrap_config,
    save_bootstrap_config,
    validate_bootstrap_config_change,
)
from .control import (
    MorpionBootstrapControl,
    apply_control_to_args,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from .cycle_dataset import (
    extract_rows_from_training_snapshot as _extract_rows_from_training_snapshot,
)
from .cycle_metadata import build_bootstrap_event
from .cycle_metadata import build_event_metadata as _build_event_metadata
from .cycle_metadata import next_metadata as _next_metadata
from .cycle_metadata import pipeline_metadata as _pipeline_metadata
from .cycle_metadata import record_no_save_cycle_event as _record_no_save_cycle_event
from .cycle_metadata import with_config_hash_metadata as _with_config_hash_metadata
from .cycle_pipeline_manifest import (
    write_pipeline_manifest_for_generation as _write_pipeline_manifest_for_generation,
)
from .cycle_runtime import build_no_save_run_state as _build_no_save_run_state
from .cycle_runtime import (
    prune_saved_generation_artifacts as _prune_saved_generation_artifacts,
)
from .cycle_runtime import resolve_active_model_bundle as _resolve_active_model_bundle
from .cycle_runtime import resolve_runtime_restore_path as _resolve_runtime_restore_path
from .cycle_runtime import resolve_tree_status as _resolve_tree_status
from .cycle_timing import save_trigger_reason as _save_trigger_reason
from .cycle_timing import should_save_progress
from .cycle_timing import timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s
from .cycle_training import train_and_select_evaluators as _train_and_select_evaluators
from .cycle_validation import (
    previous_effective_runtime_config as _previous_effective_runtime_config,
)
from .cycle_validation import reevaluate_tree_for_policy as _reevaluate_tree_for_policy
from .cycle_validation import (
    validate_dataset_family_target_args as _validate_dataset_family_target_args,
)
from .cycle_validation import validate_forced_evaluator as _validate_forced_evaluator
from .cycle_validation import validate_pipeline_mode as _validate_pipeline_mode
from .cycle_validation import (
    validate_runtime_reconfiguration as _validate_runtime_reconfiguration,
)
from .history import MorpionBootstrapHistoryRecorder
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
from .record_status import (
    resolve_frontier_status_for_cycle,
    resolve_record_status_for_cycle,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs
    from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)


def _artifact_pipeline_mode_required_error() -> ValueError:
    """Build the canonical artifact-pipeline mode requirement error."""
    return ValueError("artifact_pipeline mode required")


def _require_artifact_pipeline_mode(args: MorpionBootstrapArgs) -> None:
    """Require explicit artifact-pipeline mode for stage entrypoints."""
    if args.pipeline_mode != "artifact_pipeline":
        raise _artifact_pipeline_mode_required_error()


class MissingPipelineTreeSnapshotFileError(FileNotFoundError):
    """Raised when a pipeline dataset stage cannot find its tree snapshot."""

    @classmethod
    def from_path(cls, tree_snapshot_path: Path | None) -> MissingPipelineTreeSnapshotFileError:
        """Build one missing-tree-snapshot error with the resolved path."""
        return cls(f"Pipeline tree snapshot does not exist: {tree_snapshot_path}")


class MissingPipelineRowsFileError(FileNotFoundError):
    """Raised when a pipeline training stage cannot find its rows file."""

    @classmethod
    def from_path(cls, rows_path: Path | None) -> MissingPipelineRowsFileError:
        """Build one missing-rows error with the resolved path."""
        return cls(f"Pipeline rows file does not exist: {rows_path}")


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
    raise MissingPipelineTreeSnapshotFileError.from_path(tree_snapshot_path)


def _raise_missing_rows_file_error(rows_path: Path | None) -> NoReturn:
    """Raise the canonical training-stage missing rows file error."""
    raise MissingPipelineRowsFileError.from_path(rows_path)


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
    """Run the Phase 4 growth-only stage for artifact-pipeline mode."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    current_config = bootstrap_config_from_args(args)
    if paths.bootstrap_config_path.is_file():
        persisted_config = load_bootstrap_config(paths.bootstrap_config_path)
        validate_bootstrap_config_change(persisted_config, current_config)
    save_bootstrap_config(current_config, paths.bootstrap_config_path)
    config_hash = bootstrap_config_sha256(current_config)

    run_state = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.is_file()
        else initialize_bootstrap_run_state()
    )
    run_state = _with_config_hash_metadata(run_state, config_hash=config_hash)

    LOGGER.info("[pipeline] growth_start max_cycles=%s", max_cycles)
    cycles_run = 0
    while cycles_run < max_cycles:
        previous_generation = run_state.generation
        control = load_bootstrap_control(paths.control_path)
        effective_args = apply_control_to_args(args, control)
        run_state = _run_one_pipeline_growth_cycle(
            args=effective_args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            config_hash=config_hash,
            bootstrap_config=current_config,
        )
        save_bootstrap_run_state(run_state, paths.run_state_path)
        if run_state.generation > previous_generation:
            _prune_saved_generation_artifacts(paths)
        cycles_run += 1
    LOGGER.info(
        "[pipeline] growth_done generation=%s cycle=%s",
        run_state.generation,
        run_state.cycle_index,
    )
    return run_state


def _run_one_pipeline_growth_cycle(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None,
    config_hash: str,
    bootstrap_config: MorpionBootstrapConfig,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one growth-only artifact-pipeline cycle with memory hooks."""
    memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
    memory.log("cycle_start")
    try:
        return _run_one_pipeline_growth_cycle_impl(
            args=args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            config_hash=config_hash,
            bootstrap_config=bootstrap_config,
            now_unix_s=now_unix_s,
            memory=memory,
        )
    finally:
        log_after_cycle_gc(memory)
        memory.close()


def _run_one_pipeline_growth_cycle_impl(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None,
    config_hash: str,
    bootstrap_config: MorpionBootstrapConfig,
    now_unix_s: float | None,
    memory: MemoryDiagnostics,
) -> MorpionBootstrapRunState:
    """Run one artifact-pipeline cycle that only grows and exports artifacts."""
    cycle_started_at = time.perf_counter()
    _validate_pipeline_mode(args)
    _require_artifact_pipeline_mode(args)
    _validate_dataset_family_target_args(args)
    reevaluate_tree = _reevaluate_tree_for_policy(args.evaluator_update_policy)
    resolved_control = MorpionBootstrapControl() if control is None else control
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        bootstrap_config,
        resolved_control,
    )
    previous_effective_runtime_config = _previous_effective_runtime_config(
        run_state.metadata,
        resolved_bootstrap_config=bootstrap_config,
    )
    _validate_runtime_reconfiguration(
        previous_effective_runtime_config=previous_effective_runtime_config,
        effective_runtime_config=effective_runtime_config,
    )
    resolved_evaluators_config = args.resolved_evaluators_config()
    _validate_forced_evaluator(
        force_evaluator=resolved_control.force_evaluator,
        evaluator_names=resolved_evaluators_config.evaluators,
    )

    cycle_index = run_state.cycle_index + 1
    LOGGER.info(
        "[pipeline] growth_cycle_start cycle=%s generation=%s",
        cycle_index,
        run_state.generation,
    )
    history_recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())
    resolved_active_model = _resolve_active_model_bundle(
        paths=paths,
        latest_model_bundle_paths=run_state.latest_model_bundle_paths,
        active_evaluator_name=run_state.active_evaluator_name,
        force_evaluator=resolved_control.force_evaluator,
    )
    restore_tree_path = _resolve_runtime_restore_path(paths=paths, run_state=run_state)
    runner.load_or_create(
        restore_tree_path,
        resolved_active_model.model_bundle_path,
        effective_runtime_config,
        reevaluate_tree=reevaluate_tree,
    )
    memory.log("after_runtime_restore")
    memory.log("before_tree_growth")
    growth_started_at = time.perf_counter()
    tree_size_before_growth = runner.current_tree_size()
    runner.grow(args.max_growth_steps_per_cycle)
    growth_duration_s = time.perf_counter() - growth_started_at
    current_tree_size = runner.current_tree_size()
    memory.log("after_tree_growth")
    LOGGER.info(
        "[growth] cycle_done elapsed=%.3fs nodes_before=%s nodes_after=%s delta=%s",
        growth_duration_s,
        tree_size_before_growth,
        current_tree_size,
        current_tree_size - tree_size_before_growth,
    )
    tree_status = _resolve_tree_status(runner, current_tree_size=current_tree_size)
    frontier_status = resolve_frontier_status_for_cycle(
        snapshot=None,
        previous_frontier_status=run_state.latest_frontier_status,
    )
    current_time = time.time() if now_unix_s is None else now_unix_s
    timestamp_utc = _timestamp_utc_from_unix_s(current_time)
    save_triggered = should_save_progress(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    )
    save_reason = _save_trigger_reason(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    )

    if not save_triggered:
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info("[save] decision_done triggered=false reason=threshold_not_reached")
        LOGGER.info(
            "[timing] cycle_done growth=%.3fs training=%.3fs total_cycle=%.3fs",
            growth_duration_s,
            0.0,
            cycle_duration_s,
        )
        next_run_state = _build_no_save_run_state(
            run_state=run_state,
            resolved_active_model=resolved_active_model,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            cycle_index=cycle_index,
        )
        _record_no_save_cycle_event(
            history_recorder=history_recorder,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            frontier_status=frontier_status,
            run_state=run_state,
            next_run_state=next_run_state,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        )
        return next_run_state

    generation = run_state.generation + 1
    LOGGER.info("[save] decision_done triggered=true reason=%s", save_reason or "unknown")
    runtime_checkpoint_path = paths.runtime_checkpoint_path_for_generation(generation)
    relative_runtime_checkpoint_path: str | None = None
    save_checkpoint = getattr(runner, "save_checkpoint", None)
    if callable(save_checkpoint):
        save_checkpoint(runtime_checkpoint_path)
        if not runtime_checkpoint_path.is_file():
            raise MissingSavedBootstrapArtifactError(
                action="runner.save_checkpoint()",
                artifact_path=runtime_checkpoint_path,
            )
        relative_runtime_checkpoint_path = paths.relative_to_work_dir(
            runtime_checkpoint_path
        )
    else:
        LOGGER.info("[checkpoint] skipped reason=runner_has_no_save_checkpoint")

    tree_snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    runner.export_training_tree_snapshot(tree_snapshot_path)
    if not tree_snapshot_path.is_file():
        raise MissingSavedBootstrapArtifactError(
            action="runner.export_training_tree_snapshot()",
            artifact_path=tree_snapshot_path,
        )
    relative_tree_snapshot_path = paths.relative_to_work_dir(tree_snapshot_path)
    cycle_duration_s = time.perf_counter() - cycle_started_at
    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=relative_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=resolved_active_model.active_evaluator_name,
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        latest_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        latest_record_status=run_state.latest_record_status,
        latest_frontier_status=frontier_status,
        metadata=_next_metadata(
            run_state.metadata,
            relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        ),
    )
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=None,
            dataset_num_rows=None,
            dataset_num_samples=None,
            training_triggered=False,
            frontier_status=frontier_status,
            record_status=resolve_record_status_for_cycle(
                snapshot=None,
                previous_record_status=run_state.latest_record_status,
            ),
            metadata={
                **_build_event_metadata(
                    active_evaluator_name=next_run_state.active_evaluator_name,
                    config_hash=config_hash,
                    forced_evaluator=resolved_control.force_evaluator,
                    runtime_control=resolved_control.runtime,
                    effective_runtime_config=effective_runtime_config,
                ),
                **_pipeline_metadata(args=args),
            },
        )
    )
    _write_pipeline_manifest_for_generation(
        paths=paths,
        generation=generation,
        timestamp_utc=timestamp_utc,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        relative_tree_snapshot_path=relative_tree_snapshot_path,
        relative_rows_path=None,
        model_bundle_paths={},
        selected_evaluator_name=None,
        dataset_status="not_started",
        training_status="not_started",
        metadata=_pipeline_metadata(args=args),
    )
    LOGGER.info(
        "[pipeline] growth_cycle_done cycle=%s generation=%s saved=true elapsed=%.3fs",
        cycle_index,
        generation,
        cycle_duration_s,
    )
    return next_run_state


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
