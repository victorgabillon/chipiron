"""Shared bootstrap argument dataclass for Morpion workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


def _invalid_training_max_rows_error() -> ValueError:
    """Return the canonical training-row-limit validation error."""
    return ValueError("training_max_rows must be a non-negative integer or None.")


def _invalid_training_row_chunk_size_error() -> ValueError:
    """Return the canonical training row chunk-size validation error."""
    return ValueError("training_row_chunk_size must be a positive integer.")


def _invalid_min_available_ram_mb_error() -> ValueError:
    """Return the canonical available-RAM guard validation error."""
    return ValueError("min_available_ram_mb must be a non-negative integer or None.")


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

    def __post_init__(self) -> None:
        """Validate cross-cutting scalar controls."""
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
        if self.min_available_ram_mb is not None and (
            isinstance(self.min_available_ram_mb, bool)
            or self.min_available_ram_mb < 0
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
