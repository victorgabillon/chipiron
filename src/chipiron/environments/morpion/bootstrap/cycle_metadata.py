"""Metadata and event helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BOOTSTRAP_CONFIG_HASH_METADATA_KEY
from .control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapRuntimeControl,
    bootstrap_control_to_dict,
    bootstrap_runtime_control_to_dict,
    effective_runtime_config_sha256,
    effective_runtime_config_to_dict,
)
from .history import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
)
from .record_status import (
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
    carried_forward_morpion_frontier_status,
    default_morpion_record_status,
    morpion_bootstrap_experiment_metadata,
    resolve_record_status_for_cycle,
)
from .run_state import MorpionBootstrapRunState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_args import MorpionBootstrapArgs

RUNTIME_CHECKPOINT_METADATA_KEY = "runtime_checkpoint_path"
TRAINING_SKIPPED_REASON_METADATA_KEY = "training_skipped_reason"
EMPTY_DATASET_TRAINING_SKIPPED_REASON = "empty_dataset"


def build_bootstrap_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    tree_status: MorpionBootstrapTreeStatus,
    tree_snapshot_path: str | None,
    rows_path: str | None,
    dataset_num_rows: int | None,
    dataset_num_samples: int | None,
    training_triggered: bool,
    frontier_status: MorpionBootstrapFrontierStatus | None = None,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics] | None = None,
    model_bundle_paths: Mapping[str, str] | None = None,
    record_status: MorpionBootstrapRecordStatus | None = None,
    event_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MorpionBootstrapEvent:
    """Build one structured bootstrap history event from cycle outputs."""
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}" if event_id is None else event_id,
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc=timestamp_utc,
        tree=tree_status,
        dataset=MorpionBootstrapDatasetStatus(
            num_rows=dataset_num_rows,
            num_samples=dataset_num_samples,
        ),
        training=MorpionBootstrapTrainingStatus(triggered=training_triggered),
        record=default_morpion_record_status()
        if record_status is None
        else record_status,
        frontier=carried_forward_morpion_frontier_status(frontier_status),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=tree_snapshot_path,
            rows_path=rows_path,
            model_bundle_paths=dict(model_bundle_paths or {}),
        ),
        evaluators=dict(evaluator_metrics or {}),
        metadata=dict(metadata or {}),
    )


def build_event_metadata(
    *,
    active_evaluator_name: str | None,
    selected_evaluator_name: str | None = None,
    selection_policy: str | None = None,
    config_hash: str | None = None,
    forced_evaluator: str | None = None,
    runtime_control: MorpionBootstrapRuntimeControl | None = None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Build history metadata describing the active and selected evaluators."""
    metadata = morpion_bootstrap_experiment_metadata()
    if active_evaluator_name is not None:
        metadata["active_evaluator_name"] = active_evaluator_name
    if selected_evaluator_name is not None:
        metadata["selected_evaluator_name"] = selected_evaluator_name
        metadata["selection_policy"] = selection_policy or "lowest_final_loss"
    if config_hash is not None:
        metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    if forced_evaluator is not None:
        metadata["forced_evaluator"] = forced_evaluator
    if runtime_control is not None:
        metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
            bootstrap_runtime_control_to_dict(runtime_control)
        )
    if effective_runtime_config is not None:
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
            effective_runtime_config_to_dict(effective_runtime_config)
        )
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
            effective_runtime_config_sha256(effective_runtime_config)
        )
    if training_skipped_reason is not None:
        metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return metadata


def pipeline_metadata(
    *,
    args: MorpionBootstrapArgs,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Build pipeline-manifest metadata mirrored from the current loop config."""
    metadata: dict[str, object] = {
        "pipeline_mode": args.pipeline_mode,
        "evaluator_update_policy": args.evaluator_update_policy,
    }
    if training_skipped_reason is not None:
        metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return metadata


def bootstrap_config_hash_from_metadata(metadata: Mapping[str, object]) -> str | None:
    """Return the persisted bootstrap config hash from one metadata mapping."""
    value = metadata.get(BOOTSTRAP_CONFIG_HASH_METADATA_KEY)
    return value if isinstance(value, str) else None


def with_config_hash_metadata(
    run_state: MorpionBootstrapRunState,
    *,
    config_hash: str,
) -> MorpionBootstrapRunState:
    """Return one run state with the accepted bootstrap config hash recorded."""
    next_payload = dict(run_state.metadata)
    next_payload[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    return MorpionBootstrapRunState(
        generation=run_state.generation,
        cycle_index=run_state.cycle_index,
        latest_tree_snapshot_path=run_state.latest_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=run_state.active_evaluator_name,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        last_save_unix_s=run_state.last_save_unix_s,
        latest_runtime_checkpoint_path=run_state.latest_runtime_checkpoint_path,
        latest_record_status=run_state.latest_record_status,
        latest_frontier_status=run_state.latest_frontier_status,
        metadata=next_payload,
    )


def next_metadata(
    current_metadata: Mapping[str, object],
    *,
    relative_runtime_checkpoint_path: str | None,
    control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Return updated run metadata after one cycle boundary."""
    next_payload = dict(current_metadata)
    if relative_runtime_checkpoint_path is None:
        next_payload.pop(RUNTIME_CHECKPOINT_METADATA_KEY, None)
    else:
        next_payload[RUNTIME_CHECKPOINT_METADATA_KEY] = relative_runtime_checkpoint_path
    next_payload[BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY] = bootstrap_control_to_dict(
        control
    )
    next_payload[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
        bootstrap_runtime_control_to_dict(control.runtime)
    )
    next_payload[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
        effective_runtime_config_to_dict(effective_runtime_config)
    )
    next_payload[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
        effective_runtime_config_sha256(effective_runtime_config)
    )
    if training_skipped_reason is None:
        next_payload.pop(TRAINING_SKIPPED_REASON_METADATA_KEY, None)
    else:
        next_payload[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return next_payload


def record_no_save_cycle_event(
    *,
    history_recorder: MorpionBootstrapHistoryRecorder,
    cycle_index: int,
    timestamp_utc: str,
    tree_status: MorpionBootstrapTreeStatus,
    frontier_status: MorpionBootstrapFrontierStatus,
    run_state: MorpionBootstrapRunState,
    next_run_state: MorpionBootstrapRunState,
    resolved_control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Record the history event for a no-save cycle."""
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            tree_snapshot_path=None,
            rows_path=None,
            dataset_num_rows=None,
            dataset_num_samples=None,
            training_triggered=False,
            frontier_status=frontier_status,
            record_status=resolve_record_status_for_cycle(
                snapshot=None,
                previous_record_status=run_state.latest_record_status,
            ),
            metadata=build_event_metadata(
                active_evaluator_name=next_run_state.active_evaluator_name,
                config_hash=bootstrap_config_hash_from_metadata(run_state.metadata),
                forced_evaluator=resolved_control.force_evaluator,
                runtime_control=resolved_control.runtime,
                effective_runtime_config=effective_runtime_config,
            ),
        )
    )


__all__ = [
    "EMPTY_DATASET_TRAINING_SKIPPED_REASON",
    "RUNTIME_CHECKPOINT_METADATA_KEY",
    "TRAINING_SKIPPED_REASON_METADATA_KEY",
    "bootstrap_config_hash_from_metadata",
    "build_bootstrap_event",
    "build_event_metadata",
    "next_metadata",
    "pipeline_metadata",
    "record_no_save_cycle_event",
    "with_config_hash_metadata",
]
