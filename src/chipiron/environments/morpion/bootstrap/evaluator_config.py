"""Evaluator configuration types for the Morpion bootstrap workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
)

from .bootstrap_errors import (
    EmptyMorpionEvaluatorsConfigError,
    InconsistentMorpionEvaluatorSpecNameError,
)


def _empty_evaluator_specs() -> dict[str, MorpionEvaluatorSpec]:
    """Return a typed empty evaluator-spec mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorSpec:
    """Training spec for one named Morpion evaluator."""

    name: str
    model_type: str
    hidden_sizes: tuple[int, ...] | None
    num_epochs: int
    batch_size: int
    learning_rate: float
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    entity_max_tokens: int = 1536
    entity_input_feature_dim: int = MORPION_ENTITY_TOKEN_FEATURE_DIM
    entity_d_model: int = 64
    entity_n_head: int = 4
    entity_n_layer: int = 2
    entity_dim_feedforward: int = 256
    entity_dropout_ratio: float = 0.0
    entity_pooling: str = "value_token"
    entity_output_tanh: bool = False
    entity_use_validity_feature: bool = True
    entity_relation_schema: str | None = None
    entity_relation_type_count: int | None = None

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)
        MorpionRegressorArgs(
            model_kind=self.model_type,
            feature_subset_name=subset.name,
            feature_names=subset.feature_names,
            hidden_sizes=self.hidden_sizes,
            entity_max_tokens=self.entity_max_tokens,
            entity_input_feature_dim=self.entity_input_feature_dim,
            entity_d_model=self.entity_d_model,
            entity_n_head=self.entity_n_head,
            entity_n_layer=self.entity_n_layer,
            entity_dim_feedforward=self.entity_dim_feedforward,
            entity_dropout_ratio=self.entity_dropout_ratio,
            entity_pooling=self.entity_pooling,
            entity_output_tanh=self.entity_output_tanh,
            entity_use_validity_feature=self.entity_use_validity_feature,
            entity_relation_schema=self.entity_relation_schema,
            entity_relation_type_count=self.entity_relation_type_count,
        )

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this evaluator."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorsConfig:
    """Deterministic collection of evaluator specs for one bootstrap run."""

    evaluators: dict[str, MorpionEvaluatorSpec] = field(
        default_factory=_empty_evaluator_specs
    )

    def __post_init__(self) -> None:
        """Copy and validate the evaluator mapping eagerly."""
        copied: dict[str, MorpionEvaluatorSpec] = dict(self.evaluators)
        if not copied:
            raise EmptyMorpionEvaluatorsConfigError
        for key, spec in copied.items():
            if key != spec.name:
                raise InconsistentMorpionEvaluatorSpecNameError(key, spec.name)
        object.__setattr__(self, "evaluators", copied)


__all__ = ["MorpionEvaluatorSpec", "MorpionEvaluatorsConfig"]
