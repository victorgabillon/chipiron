"""Compatibility facade for Morpion Anemone runtime runner APIs.

Runtime implementation now lives under
``chipiron.environments.morpion.bootstrap.runtime``.
"""

from __future__ import annotations

from .runtime import checkpoint_io as _checkpoint_io
from .runtime.checkpoint_io import (
    CheckpointIoMetrics,
    _checkpoint_artifact_bytes,
    _checkpoint_node_counts,
    _is_sharded_runtime_checkpoint_path,
    _log_checkpoint_metrics,
    _metric_value,
    _pop_cached_morpion_search_checkpoint_payload_for_restore,
    cache_morpion_search_checkpoint_payload_for_restore,
    checkpoint_io_metrics_to_dict,
)
from .runtime.rollout_logging import (
    _log_latest_rollout_report,
    _log_search_rollout_config,
    _opening_expansion_config_from_rollout,
    _opening_expansion_kind_name,
    _opening_type_name,
    _rollout_no_legal_but_not_terminal,
    _rollout_path_reports,
    _rollout_path_stop_counts,
    _rollout_path_stop_reason,
)
from .runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    InvalidMorpionSearchCheckpointError,
    MorpionRegressorMasterEvaluator,
    MorpionStateToTensorConverter,
    MorpionTrainingExportProfile,
    UninitializedMorpionSearchRunnerError,
    _apply_runtime_config_to_runtime,
    _default_search_args,
    _invalid_checkpoint_payload_mapping_error,
    _new_morpion_state_checkpoint_codec,
    _runtime_config_from_search_args,
    _search_args_with_tree_branch_limit,
    apply_runtime_control_to_runner_args,
    load_morpion_evaluator_from_model_bundle,
    load_morpion_search_checkpoint_payload,
    log_morpion_checkpoint_memory_phase,
    restore_memory_logger_for_checkpoint_path,
    sharded_training_export_stats_to_dict,
    training_export_profile_to_dict,
)
from .runtime.state_eviction import (
    MorpionGrowthStateEvictionMetrics,
    _dump_live_state_parent_branch_for_checkpoint,
    _effective_growth_state_eviction_policy,
    _live_compact_state_payload_cycle_error,
    _LiveCompactStateResolver,
    _LiveEvictionPayload,
    _ParentDeltaContext,
    _phase_delta,
    _single_parent_link_for_live_delta,
)


def __getattr__(name: str) -> object:
    """Expose selected runtime checkpoint internals kept in checkpoint_io."""
    if name == "_validated_checkpoint_payload_cache":
        return _checkpoint_io._validated_checkpoint_payload_cache
    raise AttributeError(name)


__all__ = [
    "AnemoneMorpionSearchRunner",
    "AnemoneMorpionSearchRunnerArgs",
    "CheckpointIoMetrics",
    "InvalidMorpionSearchCheckpointError",
    "MorpionGrowthStateEvictionMetrics",
    "MorpionRegressorMasterEvaluator",
    "MorpionStateToTensorConverter",
    "MorpionTrainingExportProfile",
    "UninitializedMorpionSearchRunnerError",
    "_LiveCompactStateResolver",
    "_LiveEvictionPayload",
    "_ParentDeltaContext",
    "_apply_runtime_config_to_runtime",
    "_checkpoint_artifact_bytes",
    "_checkpoint_node_counts",
    "_default_search_args",
    "_dump_live_state_parent_branch_for_checkpoint",
    "_effective_growth_state_eviction_policy",
    "_invalid_checkpoint_payload_mapping_error",
    "_is_sharded_runtime_checkpoint_path",
    "_live_compact_state_payload_cycle_error",
    "_log_checkpoint_metrics",
    "_log_latest_rollout_report",
    "_log_search_rollout_config",
    "_metric_value",
    "_new_morpion_state_checkpoint_codec",
    "_opening_expansion_config_from_rollout",
    "_opening_expansion_kind_name",
    "_opening_type_name",
    "_phase_delta",
    "_pop_cached_morpion_search_checkpoint_payload_for_restore",
    "_rollout_no_legal_but_not_terminal",
    "_rollout_path_reports",
    "_rollout_path_stop_counts",
    "_rollout_path_stop_reason",
    "_runtime_config_from_search_args",
    "_search_args_with_tree_branch_limit",
    "_single_parent_link_for_live_delta",
    "apply_runtime_control_to_runner_args",
    "cache_morpion_search_checkpoint_payload_for_restore",
    "checkpoint_io_metrics_to_dict",
    "load_morpion_evaluator_from_model_bundle",
    "load_morpion_search_checkpoint_payload",
    "log_morpion_checkpoint_memory_phase",
    "restore_memory_logger_for_checkpoint_path",
    "sharded_training_export_stats_to_dict",
    "training_export_profile_to_dict",
]
