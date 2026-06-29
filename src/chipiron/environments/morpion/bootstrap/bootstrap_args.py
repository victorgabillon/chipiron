"""Shared bootstrap argument dataclass for Morpion workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .bootstrap_errors import (
    ConflictingMorpionEvaluatorConfigurationError,
    InvalidReevaluationBlendAlphaError,
)
from .config import DEFAULT_MORPION_TREE_BRANCH_LIMIT, MorpionBootstrapSearchConfig
from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
from .evaluator_family import morpion_evaluators_config_from_preset
from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    DEFAULT_MORPION_TRAINING_EXPORT_MODE,
    MorpionEvaluatorUpdatePolicy,
    MorpionPipelineMode,
    MorpionTrainingExportMode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .pv_family_targets import PvFamilyTargetPolicy

type MorpionRuntimeCheckpointFormat = Literal["json-zst", "sharded"]
type MorpionGrowthStateEvictionPolicy = Literal[
    "none", "cold_expanded", "frontier_cold", "expanded"
]
type MorpionGrowthStateEvictionPayloadMode = Literal["anchor", "delta_when_safe"]


def _invalid_training_max_rows_error() -> ValueError:
    """Return the canonical training-row-limit validation error."""
    return ValueError("training_max_rows must be a non-negative integer or None.")


def _invalid_training_row_chunk_size_error() -> ValueError:
    """Return the canonical training row chunk-size validation error."""
    return ValueError("training_row_chunk_size must be a positive integer.")


def _invalid_evaluator_diagnostics_max_rows_error() -> ValueError:
    """Return the canonical evaluator-diagnostics row-limit validation error."""
    return ValueError(
        "evaluator_diagnostics_max_rows must be a non-negative integer or None."
    )


def _invalid_growth_memory_profile_top_n_error() -> ValueError:
    """Return the canonical growth memory profile top-N validation error."""
    return ValueError("growth_memory_profile_top_n must be a positive integer.")


def _invalid_growth_memory_profile_sample_nodes_error() -> ValueError:
    """Return the canonical growth memory profile sample-size validation error."""
    return ValueError(
        "growth_memory_profile_sample_nodes must be a non-negative integer."
    )


def _invalid_growth_memory_profile_recursive_max_objects_error() -> ValueError:
    """Return the canonical recursive memory-profile object-cap error."""
    return ValueError(
        "growth_memory_profile_recursive_max_objects must be a positive integer or None."
    )


def _invalid_growth_memory_profile_recursive_context_node_cap_error() -> ValueError:
    """Return the canonical recursive context node-cap error."""
    return ValueError(
        "growth_memory_profile_recursive_context_node_cap must be a positive integer or None."
    )


def _invalid_growth_memory_profile_recursive_max_depth_error() -> ValueError:
    """Return the canonical recursive memory-profile depth-cap error."""
    return ValueError(
        "growth_memory_profile_recursive_max_depth must be a non-negative integer or None."
    )


def _invalid_growth_memory_profile_recursive_max_depth_explicit_error() -> ValueError:
    """Return the canonical recursive max-depth explicit flag error."""
    return ValueError(
        "growth_memory_profile_recursive_max_depth_explicit must be a bool."
    )


def _invalid_growth_memory_profile_recursive_events_error() -> ValueError:
    """Return the canonical recursive memory-profile event-filter error."""
    return ValueError(
        "growth_memory_profile_recursive_events must contain at least one event name."
    )


def _invalid_growth_memory_profile_recursive_complete_map_error() -> ValueError:
    """Return the canonical recursive memory-profile complete-map error."""
    return ValueError("growth_memory_profile_recursive_complete_map must be a bool.")


def _invalid_candidate_checkpoint_load_headroom_factor_error() -> ValueError:
    """Return the canonical candidate-checkpoint load headroom factor error."""
    return ValueError(
        "candidate_checkpoint_load_headroom_factor must be a non-negative number."
    )


def _invalid_candidate_checkpoint_load_min_headroom_mb_error() -> ValueError:
    """Return the canonical candidate-checkpoint load minimum headroom error."""
    return ValueError(
        "candidate_checkpoint_load_min_headroom_mb must be a non-negative integer."
    )


def _invalid_min_available_ram_mb_error() -> ValueError:
    """Return the canonical available-RAM guard validation error."""
    return ValueError("min_available_ram_mb must be a non-negative integer or None.")


def _invalid_runtime_checkpoint_format_error() -> ValueError:
    """Return the canonical runtime checkpoint format validation error."""
    return ValueError("runtime_checkpoint_format must be 'json-zst' or 'sharded'.")


def _invalid_diagnostic_stop_after_growth_error() -> TypeError:
    """Return the canonical diagnostic stop flag validation error."""
    return TypeError("diagnostic_stop_after_growth must be a bool.")


def _invalid_growth_state_eviction_policy_error() -> ValueError:
    """Return the canonical growth state-eviction policy validation error."""
    return ValueError(
        "growth_state_eviction_policy must be 'none', 'cold_expanded', "
        "or 'frontier_cold'."
    )


def _normalize_growth_state_eviction_policy(
    policy: MorpionGrowthStateEvictionPolicy,
) -> MorpionGrowthStateEvictionPolicy:
    """Normalize legacy state-eviction policy spelling."""
    if policy == "expanded":
        return "cold_expanded"
    return policy


def _invalid_growth_state_eviction_recent_window_error() -> ValueError:
    """Return the canonical growth state-eviction recent-window error."""
    return ValueError(
        "growth_state_eviction_recent_window must be a non-negative integer."
    )


def _invalid_growth_state_rematerialization_cache_size_error() -> ValueError:
    """Return the canonical rematerialization cache-size error."""
    return ValueError(
        "growth_state_rematerialization_cache_size must be a non-negative integer."
    )


def _invalid_growth_state_eviction_scan_interval_steps_error() -> ValueError:
    """Return the canonical growth state-eviction scan interval error."""
    return ValueError(
        "growth_state_eviction_scan_interval_steps must be a positive integer."
    )


def _invalid_growth_state_eviction_scan_node_limit_error() -> ValueError:
    """Return the canonical growth state-eviction scan node-limit error."""
    return ValueError(
        "growth_state_eviction_scan_node_limit must be a positive integer."
    )


def _invalid_growth_state_eviction_payload_mode_error() -> ValueError:
    """Return the canonical growth state-eviction payload-mode error."""
    return ValueError(
        "growth_state_eviction_payload_mode must be 'anchor' or 'delta_when_safe'."
    )


def _invalid_growth_state_eviction_delta_chain_max_depth_error() -> ValueError:
    """Return the canonical growth state-eviction delta-chain depth error."""
    return ValueError(
        "growth_state_eviction_delta_chain_max_depth must be a positive integer."
    )


def _invalid_growth_additional_branch_budget_error() -> ValueError:
    """Return the canonical additional branch-budget validation error."""
    return ValueError(
        "growth_additional_branch_budget must be a positive integer or None."
    )


def _invalid_growth_save_and_exit_error() -> TypeError:
    """Return the canonical grow-save-exit flag validation error."""
    return TypeError("growth_save_and_exit must be a bool.")


def _invalid_growth_skip_training_export_error() -> TypeError:
    """Return the canonical skip-training-export flag validation error."""
    return TypeError("growth_skip_training_export must be a bool.")


@dataclass(frozen=True, slots=True)
class MorpionBootstrapArgs:
    """Top-level arguments for the restartable Morpion bootstrap loop."""

    work_dir: str | Path
    max_growth_steps_per_cycle: int = 1000
    save_after_tree_growth_factor: float = 2.0
    save_after_seconds: float = 3600.0
    require_exact_or_terminal: bool = False
    min_depth: int | None = None
    min_visit_count: int | None = None
    max_rows: int | None = None
    use_backed_up_value: bool = True
    dataset_family_target_policy: PvFamilyTargetPolicy = "none"
    dataset_family_prediction_blend: float = 0.25
    memory_diagnostics: bool = False
    memory_diagnostics_gc_growth: bool = False
    memory_diagnostics_tracemalloc: bool = False
    memory_diagnostics_torch_tensors: bool = False
    memory_diagnostics_referrers: bool = False
    memory_diagnostics_referrer_type_patterns: tuple[str, ...] = ()
    memory_diagnostics_referrer_max_objects_per_type: int = 2
    memory_diagnostics_referrer_max_depth: int = 2
    memory_diagnostics_top_n: int = 20
    min_available_ram_mb: int | None = None
    tree_branch_limit: int = DEFAULT_MORPION_TREE_BRANCH_LIMIT
    reevaluation_blend_alpha: float = 1.0
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    validation_fraction: float = 0.2
    validation_seed: int = 0
    model_kind: str = "linear"
    hidden_dim: int | None = None
    evaluator_update_policy: MorpionEvaluatorUpdatePolicy = (
        DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY
    )
    pipeline_mode: MorpionPipelineMode = DEFAULT_MORPION_PIPELINE_MODE
    training_export_mode: MorpionTrainingExportMode = (
        DEFAULT_MORPION_TRAINING_EXPORT_MODE
    )
    search: MorpionBootstrapSearchConfig = field(
        default_factory=MorpionBootstrapSearchConfig
    )
    evaluators_config: MorpionEvaluatorsConfig | None = None
    evaluator_family_preset: str | None = None
    training_evaluator_names: tuple[str, ...] | None = None
    training_max_rows: int | None = None
    training_row_chunk_size: int = 8192
    skip_evaluator_diagnostics: bool = False
    evaluator_diagnostics_max_rows: int | None = 60
    growth_memory_profile: bool = False
    growth_memory_profile_top_n: int = 20
    growth_memory_profile_sample_nodes: int = 2000
    growth_memory_profile_recursive: bool = False
    growth_memory_profile_recursive_max_objects: int | None = None
    growth_memory_profile_recursive_max_depth: int | None = None
    growth_memory_profile_recursive_max_depth_explicit: bool = False
    growth_memory_profile_recursive_context_node_cap: int | None = None
    growth_memory_profile_recursive_events: tuple[str, ...] = ("after_checkpoint_load",)
    growth_memory_profile_recursive_complete_map: bool = False
    candidate_checkpoint_load_headroom_factor: float = 60.0
    candidate_checkpoint_load_min_headroom_mb: int = 512
    runtime_checkpoint_format: MorpionRuntimeCheckpointFormat = "json-zst"
    diagnostic_stop_after_growth: bool = False
    growth_additional_branch_budget: int | None = None
    growth_save_and_exit: bool = False
    growth_skip_training_export: bool = False
    growth_state_eviction_policy: MorpionGrowthStateEvictionPolicy = "none"
    growth_state_eviction_recent_window: int = 1000
    growth_state_rematerialization_cache_size: int = 10000
    growth_state_eviction_scan_interval_steps: int = 100
    growth_state_eviction_scan_node_limit: int = 5000
    growth_state_eviction_payload_mode: MorpionGrowthStateEvictionPayloadMode = "anchor"
    growth_state_eviction_delta_chain_max_depth: int = 32

    def __post_init__(self) -> None:
        """Validate cross-cutting scalar controls."""
        if not isinstance(self.diagnostic_stop_after_growth, bool):
            raise _invalid_diagnostic_stop_after_growth_error()
        if self.growth_additional_branch_budget is not None and (
            isinstance(self.growth_additional_branch_budget, bool)
            or self.growth_additional_branch_budget <= 0
        ):
            raise _invalid_growth_additional_branch_budget_error()
        if not isinstance(self.growth_save_and_exit, bool):
            raise _invalid_growth_save_and_exit_error()
        if not isinstance(self.growth_skip_training_export, bool):
            raise _invalid_growth_skip_training_export_error()
        if self.growth_state_eviction_policy not in {
            "none",
            "cold_expanded",
            "frontier_cold",
            "expanded",
        }:
            raise _invalid_growth_state_eviction_policy_error()
        object.__setattr__(
            self,
            "growth_state_eviction_policy",
            _normalize_growth_state_eviction_policy(self.growth_state_eviction_policy),
        )
        if (
            isinstance(self.growth_state_eviction_recent_window, bool)
            or self.growth_state_eviction_recent_window < 0
        ):
            raise _invalid_growth_state_eviction_recent_window_error()
        if (
            isinstance(self.growth_state_rematerialization_cache_size, bool)
            or self.growth_state_rematerialization_cache_size < 0
        ):
            raise _invalid_growth_state_rematerialization_cache_size_error()
        if (
            isinstance(self.growth_state_eviction_scan_interval_steps, bool)
            or self.growth_state_eviction_scan_interval_steps <= 0
        ):
            raise _invalid_growth_state_eviction_scan_interval_steps_error()
        if (
            isinstance(self.growth_state_eviction_scan_node_limit, bool)
            or self.growth_state_eviction_scan_node_limit <= 0
        ):
            raise _invalid_growth_state_eviction_scan_node_limit_error()
        if self.growth_state_eviction_payload_mode not in {
            "anchor",
            "delta_when_safe",
        }:
            raise _invalid_growth_state_eviction_payload_mode_error()
        if (
            isinstance(self.growth_state_eviction_delta_chain_max_depth, bool)
            or self.growth_state_eviction_delta_chain_max_depth <= 0
        ):
            raise _invalid_growth_state_eviction_delta_chain_max_depth_error()
        if isinstance(self.reevaluation_blend_alpha, bool) or not (
            0.0 <= self.reevaluation_blend_alpha <= 1.0
        ):
            raise InvalidReevaluationBlendAlphaError
        if self.training_max_rows is not None and (
            isinstance(self.training_max_rows, bool) or self.training_max_rows < 0
        ):
            raise _invalid_training_max_rows_error()
        if (
            isinstance(self.training_row_chunk_size, bool)
            or self.training_row_chunk_size <= 0
        ):
            raise _invalid_training_row_chunk_size_error()
        if self.evaluator_diagnostics_max_rows is not None and (
            isinstance(self.evaluator_diagnostics_max_rows, bool)
            or self.evaluator_diagnostics_max_rows < 0
        ):
            raise _invalid_evaluator_diagnostics_max_rows_error()
        if (
            isinstance(self.growth_memory_profile_top_n, bool)
            or self.growth_memory_profile_top_n <= 0
        ):
            raise _invalid_growth_memory_profile_top_n_error()
        if (
            isinstance(self.growth_memory_profile_sample_nodes, bool)
            or self.growth_memory_profile_sample_nodes < 0
        ):
            raise _invalid_growth_memory_profile_sample_nodes_error()
        if self.growth_memory_profile_recursive_max_objects is not None and (
            isinstance(self.growth_memory_profile_recursive_max_objects, bool)
            or self.growth_memory_profile_recursive_max_objects <= 0
        ):
            raise _invalid_growth_memory_profile_recursive_max_objects_error()
        if self.growth_memory_profile_recursive_max_depth is not None and (
            isinstance(self.growth_memory_profile_recursive_max_depth, bool)
            or self.growth_memory_profile_recursive_max_depth < 0
        ):
            raise _invalid_growth_memory_profile_recursive_max_depth_error()
        if not isinstance(
            self.growth_memory_profile_recursive_max_depth_explicit, bool
        ):
            raise _invalid_growth_memory_profile_recursive_max_depth_explicit_error()
        if self.growth_memory_profile_recursive_context_node_cap is not None and (
            isinstance(self.growth_memory_profile_recursive_context_node_cap, bool)
            or self.growth_memory_profile_recursive_context_node_cap <= 0
        ):
            raise _invalid_growth_memory_profile_recursive_context_node_cap_error()
        if not self.growth_memory_profile_recursive_events or any(
            not isinstance(event, str) or not event
            for event in self.growth_memory_profile_recursive_events
        ):
            raise _invalid_growth_memory_profile_recursive_events_error()
        if not isinstance(self.growth_memory_profile_recursive_complete_map, bool):
            raise _invalid_growth_memory_profile_recursive_complete_map_error()
        if (
            isinstance(self.candidate_checkpoint_load_headroom_factor, bool)
            or self.candidate_checkpoint_load_headroom_factor < 0
        ):
            raise _invalid_candidate_checkpoint_load_headroom_factor_error()
        if (
            isinstance(self.candidate_checkpoint_load_min_headroom_mb, bool)
            or self.candidate_checkpoint_load_min_headroom_mb < 0
        ):
            raise _invalid_candidate_checkpoint_load_min_headroom_mb_error()
        if self.runtime_checkpoint_format not in {"json-zst", "sharded"}:
            raise _invalid_runtime_checkpoint_format_error()
        if self.min_available_ram_mb is not None and (
            isinstance(self.min_available_ram_mb, bool) or self.min_available_ram_mb < 0
        ):
            raise _invalid_min_available_ram_mb_error()

    def resolved_evaluators_config(self) -> MorpionEvaluatorsConfig:
        """Resolve the explicit or legacy single-evaluator config."""
        if (
            self.evaluators_config is not None
            and self.evaluator_family_preset is not None
        ):
            conflict_error = ConflictingMorpionEvaluatorConfigurationError()
            raise conflict_error
        if self.evaluators_config is not None:
            return self.evaluators_config
        if self.evaluator_family_preset is not None:
            return morpion_evaluators_config_from_preset(self.evaluator_family_preset)
        hidden_sizes = None if self.hidden_dim is None else (self.hidden_dim,)
        default_spec = MorpionEvaluatorSpec(
            name="default",
            model_type=self.model_kind,
            hidden_sizes=hidden_sizes,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        return MorpionEvaluatorsConfig(evaluators={"default": default_spec})


__all__ = ["MorpionBootstrapArgs"]
