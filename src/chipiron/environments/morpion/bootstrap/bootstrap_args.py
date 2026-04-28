"""Shared bootstrap argument dataclass for Morpion workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .bootstrap_errors import ConflictingMorpionEvaluatorConfigurationError
from .config import DEFAULT_MORPION_TREE_BRANCH_LIMIT
from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
from .evaluator_family import morpion_evaluators_config_from_preset
from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    MorpionEvaluatorUpdatePolicy,
    MorpionPipelineMode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .pv_family_targets import PvFamilyTargetPolicy


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
    tree_branch_limit: int = DEFAULT_MORPION_TREE_BRANCH_LIMIT
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    hidden_dim: int | None = None
    evaluator_update_policy: MorpionEvaluatorUpdatePolicy = (
        DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY
    )
    pipeline_mode: MorpionPipelineMode = DEFAULT_MORPION_PIPELINE_MODE
    evaluators_config: MorpionEvaluatorsConfig | None = None
    evaluator_family_preset: str | None = None

    def resolved_evaluators_config(self) -> MorpionEvaluatorsConfig:
        """Resolve the explicit or legacy single-evaluator config."""
        if (
            self.evaluators_config is not None
            and self.evaluator_family_preset is not None
        ):
            raise ConflictingMorpionEvaluatorConfigurationError
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
