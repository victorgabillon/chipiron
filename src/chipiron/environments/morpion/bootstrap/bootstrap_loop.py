"""Artifact-driven bootstrap loop for Morpion self-training."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import cast

from .bootstrap_args import MorpionBootstrapArgs
from .bootstrap_errors import (
    EmptyMorpionEvaluatorsConfigError,
    IncompatibleMorpionResumeArtifactError,
    InconsistentMorpionEvaluatorSpecNameError,
    MissingActiveMorpionEvaluatorError,
    MissingForcedMorpionEvaluatorBundleError,
    MissingSavedBootstrapArtifactError,
    NoSelectableMorpionEvaluatorError,
    UnknownActiveMorpionEvaluatorError,
    UnknownForcedMorpionEvaluatorError,
    UnsupportedMorpionRuntimeReconfigurationError,
)
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
    MorpionBootstrapEffectiveRuntimeConfig,
    apply_control_to_args,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from .cycle_dataset import (
    build_and_save_dataset_for_generation as _build_and_save_dataset_for_generation,
)
from .cycle_metadata import (
    EMPTY_DATASET_TRAINING_SKIPPED_REASON,
    RUNTIME_CHECKPOINT_METADATA_KEY,
    TRAINING_SKIPPED_REASON_METADATA_KEY,
    build_bootstrap_event,
)
from .cycle_metadata import (
    bootstrap_config_hash_from_metadata as _bootstrap_config_hash_from_metadata,
)
from .cycle_metadata import (
    build_event_metadata as _build_event_metadata,
)
from .cycle_metadata import (
    next_metadata as _next_metadata,
)
from .cycle_metadata import (
    pipeline_metadata as _pipeline_metadata,
)
from .cycle_metadata import (
    record_no_save_cycle_event as _record_no_save_cycle_event,
)
from .cycle_metadata import (
    with_config_hash_metadata as _with_config_hash_metadata,
)
from .cycle_pipeline_manifest import (
    write_pipeline_active_model as _write_pipeline_active_model,
)
from .cycle_pipeline_manifest import (
    write_pipeline_manifest_for_generation as _write_pipeline_manifest_for_generation,
)
from .cycle_runtime import (
    build_no_save_run_state as _build_no_save_run_state,
)
from .cycle_runtime import (
    prune_saved_generation_artifacts as _prune_saved_generation_artifacts,
)
from .cycle_runtime import (
    resolve_active_model_bundle as _resolve_active_model_bundle,
)
from .cycle_runtime import (
    resolve_runtime_restore_path as _resolve_runtime_restore_path,
)
from .cycle_runtime import (
    resolve_tree_status as _resolve_tree_status,
)
from .cycle_timing import (
    save_trigger_reason as _save_trigger_reason,
)
from .cycle_timing import (
    should_save_progress,
)
from .cycle_timing import (
    timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s,
)
from .cycle_training import select_active_evaluator_name
from .cycle_training import (
    train_and_select_evaluators as _train_and_select_evaluators,
)
from .cycle_validation import (
    previous_effective_runtime_config as _previous_effective_runtime_config,
)
from .cycle_validation import reevaluate_tree_for_policy as _reevaluate_tree_for_policy
from .cycle_validation import (
    require_single_process_mode as _require_single_process_mode,
)
from .cycle_validation import (
    validate_dataset_family_target_args as _validate_dataset_family_target_args,
)
from .cycle_validation import validate_forced_evaluator as _validate_forced_evaluator
from .cycle_validation import validate_pipeline_mode as _validate_pipeline_mode
from .cycle_validation import (
    validate_runtime_reconfiguration as _validate_runtime_reconfiguration,
)
from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
from .history import (
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
)
from .memory_diagnostics import MemoryDiagnostics
from .pipeline_artifacts import save_pipeline_training_status_file
from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    MorpionEvaluatorUpdatePolicy,
    MorpionPipelineMode,
)
from .record_status import (
    persist_certified_leaderboard_candidates,
    resolve_frontier_status_for_cycle,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)


def _configure_linoo_selection_artifact_for_growth(
    *,
    runner: object,
    paths: MorpionBootstrapPaths,
    cycle_index: int,
    generation: int,
) -> None:
    """Configure latest Linoo table persistence when the runner supports it."""
    configure = getattr(runner, "configure_linoo_selection_table_artifact", None)
    if not callable(configure):
        return
    configure(
        path=paths.latest_linoo_selection_table_path,
        cycle_index=cycle_index,
        generation=generation,
    )


def run_one_bootstrap_cycle(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None = None,
    bootstrap_config: MorpionBootstrapConfig | None = None,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one grow/export/train/save bootstrap cycle.

    This helper calls ``runner.load_or_create(...)`` on every cycle, so runner
    implementations should support reload/restart-style semantics from the
    latest saved artifacts.
    """
    memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
    memory.log("cycle_start")
    try:
        return _run_one_bootstrap_cycle_impl(
            args=args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            bootstrap_config=bootstrap_config,
            now_unix_s=now_unix_s,
            memory=memory,
        )
    finally:
        log_after_cycle_gc(memory)
        memory.close()


def _run_one_bootstrap_cycle_impl(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    memory: MemoryDiagnostics,
    control: MorpionBootstrapControl | None = None,
    bootstrap_config: MorpionBootstrapConfig | None = None,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one grow/export/train/save bootstrap cycle with memory hooks."""
    cycle_started_at = time.perf_counter()
    _validate_pipeline_mode(args)
    _require_single_process_mode(args)
    reevaluate_tree = _reevaluate_tree_for_policy(args.evaluator_update_policy)
    paths.ensure_directories()
    resolved_control = MorpionBootstrapControl() if control is None else control
    _validate_dataset_family_target_args(args)
    resolved_bootstrap_config = (
        bootstrap_config_from_args(args)
        if bootstrap_config is None
        else bootstrap_config
    )
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        resolved_bootstrap_config,
        resolved_control,
    )
    previous_effective_runtime_config = _previous_effective_runtime_config(
        run_state.metadata,
        resolved_bootstrap_config=resolved_bootstrap_config,
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
        "[cycle] start cycle=%s generation=%s",
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
    LOGGER.info(
        "[runtime] evaluator_update_policy policy=%s reevaluate_tree=%s",
        args.evaluator_update_policy,
        reevaluate_tree,
    )

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
    _configure_linoo_selection_artifact_for_growth(
        runner=runner,
        paths=paths,
        cycle_index=cycle_index,
        generation=run_state.generation,
    )
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
    tree_status = _resolve_tree_status(
        runner,
        current_tree_size=current_tree_size,
    )
    frontier_status = resolve_frontier_status_for_cycle(
        snapshot=None,
        previous_frontier_status=run_state.latest_frontier_status,
    )
    current_time = time.time() if now_unix_s is None else now_unix_s
    timestamp_utc = _timestamp_utc_from_unix_s(current_time)

    LOGGER.info("[save] decision_start")
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
        LOGGER.info("[save] skipped reason=threshold_not_reached")
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
        LOGGER.info(
            "[cycle] done cycle=%s generation=%s saved=false training=false",
            cycle_index,
            next_run_state.generation,
        )
        return next_run_state

    generation = run_state.generation + 1
    LOGGER.info(
        "[save] decision_done triggered=true reason=%s",
        save_reason or "unknown",
    )
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

    dataset_result = _build_and_save_dataset_for_generation(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        generation=generation,
        memory=memory,
    )
    snapshot = dataset_result.snapshot
    rows = dataset_result.rows
    record_status = dataset_result.record_status
    frontier_status = dataset_result.frontier_status
    relative_tree_snapshot_path = dataset_result.relative_tree_snapshot_path
    relative_rows_path = dataset_result.relative_rows_path
    dataset_elapsed_s = dataset_result.dataset_elapsed_s
    num_rows = dataset_result.num_rows
    rows_path = paths.rows_path_for_generation(generation)
    del dataset_result

    if num_rows == 0:
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info("[train] skipped reason=%s", EMPTY_DATASET_TRAINING_SKIPPED_REASON)
        LOGGER.info(
            "[timing] cycle_done growth=%.3fs dataset=%.3fs training=%.3fs total_cycle=%.3fs",
            growth_duration_s,
            dataset_elapsed_s,
            0.0,
            cycle_duration_s,
        )
        preserved_model_bundle_paths = (
            None
            if run_state.latest_model_bundle_paths is None
            else dict(run_state.latest_model_bundle_paths)
        )
        next_metadata = _next_metadata(
            run_state.metadata,
            relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            training_skipped_reason=EMPTY_DATASET_TRAINING_SKIPPED_REASON,
        )
        next_run_state = MorpionBootstrapRunState(
            generation=generation,
            cycle_index=cycle_index,
            latest_tree_snapshot_path=relative_tree_snapshot_path,
            latest_rows_path=relative_rows_path,
            latest_model_bundle_paths=preserved_model_bundle_paths,
            active_evaluator_name=run_state.active_evaluator_name,
            tree_size_at_last_save=current_tree_size,
            last_save_unix_s=current_time,
            latest_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            latest_record_status=record_status,
            latest_frontier_status=frontier_status,
            metadata=next_metadata,
        )
        _write_pipeline_manifest_for_generation(
            paths=paths,
            generation=generation,
            timestamp_utc=timestamp_utc,
            relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            relative_tree_snapshot_path=relative_tree_snapshot_path,
            relative_rows_path=relative_rows_path,
            model_bundle_paths={},
            selected_evaluator_name=None,
            dataset_status="done",
            training_status="not_started",
            metadata=_pipeline_metadata(
                args=args,
                training_skipped_reason=EMPTY_DATASET_TRAINING_SKIPPED_REASON,
            ),
        )
        history_recorder.record(
            build_bootstrap_event(
                cycle_index=cycle_index,
                generation=next_run_state.generation,
                timestamp_utc=timestamp_utc,
                tree_status=tree_status,
                tree_snapshot_path=relative_tree_snapshot_path,
                rows_path=relative_rows_path,
                dataset_num_rows=num_rows,
                dataset_num_samples=num_rows,
                training_triggered=False,
                frontier_status=frontier_status,
                record_status=record_status,
                metadata=_build_event_metadata(
                    active_evaluator_name=next_run_state.active_evaluator_name,
                    config_hash=_bootstrap_config_hash_from_metadata(
                        run_state.metadata
                    ),
                    forced_evaluator=resolved_control.force_evaluator,
                    runtime_control=resolved_control.runtime,
                    effective_runtime_config=effective_runtime_config,
                    training_skipped_reason=EMPTY_DATASET_TRAINING_SKIPPED_REASON,
                ),
            )
        )
        LOGGER.info(
            "[cycle] done cycle=%s generation=%s saved=true training=false",
            cycle_index,
            next_run_state.generation,
        )
        del rows
        del snapshot
        return next_run_state

    training_result = _train_and_select_evaluators(
        args=args,
        paths=paths,
        run_state=run_state,
        rows=rows,
        rows_path=rows_path,
        generation=generation,
        timestamp_utc=timestamp_utc,
        resolved_evaluators_config=resolved_evaluators_config,
        resolved_control=resolved_control,
        memory=memory,
    )
    evaluator_metrics = training_result.evaluator_metrics
    evaluator_results = training_result.evaluator_results
    model_bundle_paths = training_result.model_bundle_paths
    selected_evaluator_name = training_result.selected_evaluator_name
    selection_policy = training_result.selection_policy
    training_duration_s = training_result.training_duration_s
    del training_result
    cycle_duration_s = time.perf_counter() - cycle_started_at
    LOGGER.info(
        "[leaderboard] persist_start generation=%s cycle=%s", generation, cycle_index
    )
    leaderboard_started_at = time.perf_counter()
    try:
        persist_certified_leaderboard_candidates(
            snapshot=snapshot,
            run_work_dir=paths.work_dir,
            generation=generation,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
        )
    finally:
        LOGGER.info(
            "[leaderboard] persist_done elapsed=%.3fs",
            time.perf_counter() - leaderboard_started_at,
        )
    LOGGER.info(
        "[timing] cycle_done growth=%.3fs dataset=%.3fs training=%.3fs total_cycle=%.3fs",
        growth_duration_s,
        dataset_elapsed_s,
        training_duration_s,
        cycle_duration_s,
    )

    next_metadata = _next_metadata(
        run_state.metadata,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        control=resolved_control,
        effective_runtime_config=effective_runtime_config,
    )
    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=relative_tree_snapshot_path,
        latest_rows_path=relative_rows_path,
        latest_model_bundle_paths=model_bundle_paths,
        active_evaluator_name=selected_evaluator_name,
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        latest_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        latest_record_status=record_status,
        latest_frontier_status=frontier_status,
        metadata=next_metadata,
    )
    _write_pipeline_manifest_for_generation(
        paths=paths,
        generation=generation,
        timestamp_utc=timestamp_utc,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        relative_tree_snapshot_path=relative_tree_snapshot_path,
        relative_rows_path=relative_rows_path,
        model_bundle_paths=model_bundle_paths,
        selected_evaluator_name=selected_evaluator_name,
        dataset_status="done",
        training_status="done",
        metadata=_pipeline_metadata(args=args),
    )
    _write_pipeline_active_model(
        paths=paths,
        generation=generation,
        selected_evaluator_name=selected_evaluator_name,
        model_bundle_paths=model_bundle_paths,
        timestamp_utc=timestamp_utc,
        selection_policy=selection_policy,
    )
    save_pipeline_training_status_file(
        generation=generation,
        training_status="done",
        updated_at_utc=timestamp_utc,
        metadata=_pipeline_metadata(args=args),
        selected_evaluator_name=selected_evaluator_name,
        selection_policy=selection_policy,
        evaluator_results=evaluator_results,
        path=paths.pipeline_training_status_path_for_generation(generation),
    )
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=relative_rows_path,
            dataset_num_rows=len(rows.rows),
            dataset_num_samples=len(rows.rows),
            training_triggered=True,
            frontier_status=frontier_status,
            evaluator_metrics=evaluator_metrics,
            model_bundle_paths=model_bundle_paths,
            record_status=record_status,
            metadata=_build_event_metadata(
                active_evaluator_name=selected_evaluator_name,
                selected_evaluator_name=selected_evaluator_name,
                selection_policy=selection_policy,
                config_hash=_bootstrap_config_hash_from_metadata(run_state.metadata),
                forced_evaluator=resolved_control.force_evaluator,
                runtime_control=resolved_control.runtime,
                effective_runtime_config=effective_runtime_config,
            ),
        )
    )
    LOGGER.info(
        "[cycle] done cycle=%s generation=%s saved=true training=true selected_evaluator=%s",
        cycle_index,
        next_run_state.generation,
        selected_evaluator_name,
    )
    del rows
    del snapshot
    return next_run_state


def run_morpion_bootstrap_loop(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_cycles: int | None = None,
) -> MorpionBootstrapRunState:
    """Run the Morpion bootstrap loop for a bounded number of cycles or forever."""
    _validate_pipeline_mode(args)
    _require_single_process_mode(args)
    return _run_bootstrap_loop_impl(args, runner, max_cycles=max_cycles)


def _run_bootstrap_loop_impl(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_cycles: int | None,
) -> MorpionBootstrapRunState:
    """Run the shared bootstrap loop body for one validated execution mode."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    current_config = bootstrap_config_from_args(args)
    if paths.bootstrap_config_path.is_file():
        persisted_config = load_bootstrap_config(paths.bootstrap_config_path)
        validate_bootstrap_config_change(persisted_config, current_config)
    save_bootstrap_config(current_config, paths.bootstrap_config_path)
    config_hash = bootstrap_config_sha256(current_config)

    if paths.run_state_path.is_file():
        run_state = load_bootstrap_run_state(paths.run_state_path)
    else:
        run_state = initialize_bootstrap_run_state()
    run_state = _with_config_hash_metadata(run_state, config_hash=config_hash)

    cycles_run = 0
    LOGGER.info(
        "[launcher] loop_start work_dir=%s max_cycles=%s",
        str(paths.work_dir),
        "none" if max_cycles is None else str(max_cycles),
    )
    while max_cycles is None or cycles_run < max_cycles:
        next_cycle_index = run_state.cycle_index + 1
        previous_generation = run_state.generation
        runner_tree_size, runner_expanded_nodes = _cycle_start_tree_metrics(
            runner=runner,
            run_state=run_state,
        )
        LOGGER.info(
            "[cycle] prepare cycle=%s generation=%s tree_size=%s expanded_nodes=%s",
            next_cycle_index,
            run_state.generation,
            runner_tree_size,
            runner_expanded_nodes,
        )
        control = load_bootstrap_control(paths.control_path)
        effective_args = apply_control_to_args(args, control)
        run_state = run_one_bootstrap_cycle(
            args=effective_args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            bootstrap_config=current_config,
        )
        save_bootstrap_run_state(run_state, paths.run_state_path)
        if run_state.generation > previous_generation:
            _prune_saved_generation_artifacts(paths)
        cycles_run += 1

    LOGGER.info("[launcher] loop_done cycles_run=%s", cycles_run)
    return run_state


def _cycle_start_tree_metrics(
    *,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
) -> tuple[int | None, int | None]:
    """Return best-effort live metrics for cycle-start logging."""
    tree_size: int | None = None
    expanded_nodes: int | None = None
    current_tree_size = getattr(runner, "current_tree_size", None)
    if callable(current_tree_size):
        try:
            raw_tree_size = current_tree_size()
        except Exception:
            raw_tree_size = None
        if isinstance(raw_tree_size, int):
            tree_size = raw_tree_size
    current_tree_status = getattr(runner, "current_tree_status", None)
    if callable(current_tree_status):
        try:
            raw_tree_status = current_tree_status()
        except Exception:
            raw_tree_status = None
        if isinstance(raw_tree_status, MorpionBootstrapTreeStatus):
            expanded_nodes = raw_tree_status.num_expanded_nodes
            if tree_size is None:
                tree_size = raw_tree_status.num_nodes
        elif isinstance(raw_tree_status, Mapping):
            tree_status_mapping = cast("Mapping[str, object]", raw_tree_status)
            raw_expanded_nodes = tree_status_mapping.get("num_expanded_nodes")
            if isinstance(raw_expanded_nodes, int):
                expanded_nodes = raw_expanded_nodes
            raw_num_nodes = tree_status_mapping.get("num_nodes")
            if tree_size is None and isinstance(raw_num_nodes, int):
                tree_size = raw_num_nodes
    if tree_size is None:
        tree_size = run_state.tree_size_at_last_save
    return tree_size, expanded_nodes


__all__ = [
    "DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY",
    "DEFAULT_MORPION_PIPELINE_MODE",
    "EMPTY_DATASET_TRAINING_SKIPPED_REASON",
    "RUNTIME_CHECKPOINT_METADATA_KEY",
    "TRAINING_SKIPPED_REASON_METADATA_KEY",
    "EmptyMorpionEvaluatorsConfigError",
    "IncompatibleMorpionResumeArtifactError",
    "InconsistentMorpionEvaluatorSpecNameError",
    "MissingActiveMorpionEvaluatorError",
    "MissingForcedMorpionEvaluatorBundleError",
    "MorpionBootstrapArgs",
    "MorpionBootstrapEffectiveRuntimeConfig",
    "MorpionBootstrapPaths",
    "MorpionBootstrapRecordStatus",
    "MorpionEvaluatorMetrics",
    "MorpionEvaluatorSpec",
    "MorpionEvaluatorUpdatePolicy",
    "MorpionEvaluatorsConfig",
    "MorpionPipelineMode",
    "MorpionSearchRunner",
    "NoSelectableMorpionEvaluatorError",
    "UnknownActiveMorpionEvaluatorError",
    "UnknownForcedMorpionEvaluatorError",
    "UnsupportedMorpionRuntimeReconfigurationError",
    "_reevaluate_tree_for_policy",
    "_validate_pipeline_mode",
    "build_bootstrap_event",
    "run_morpion_bootstrap_loop",
    "run_one_bootstrap_cycle",
    "select_active_evaluator_name",
    "should_save_progress",
]
