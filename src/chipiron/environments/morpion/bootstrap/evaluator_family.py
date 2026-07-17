"""Canonical Morpion evaluator-family presets for bootstrap experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_MODEL_KIND,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MorpionFeatureSubset,
    morpion_feature_subset_from_name,
)

from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec

CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET: Final[str] = "canonical_8_linear_mlp_subsets"
CANONICAL_LINEAR_MLP_ENTITY_AND_RELATIONAL_TRANSFORMERS_SMALL_MORPION_EVALUATOR_FAMILY_PRESET: Final[
    str
] = "canonical_linear_mlp_entity_and_relational_transformers_small"
CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET: Final[
    str
] = "canonical_linear_mlp_entity_transformer_small"

_CANONICAL_MORPION_EVALUATOR_SPECS: Final[
    tuple[tuple[str, str, str, tuple[int, ...] | None], ...]
] = (
    ("linear_5", "linear", "handcrafted_5_core", None),
    ("mlp_5", "mlp", "handcrafted_5_core", (5, 10, 10)),
    ("linear_10", "linear", "handcrafted_10_core", None),
    ("mlp_10", "mlp", "handcrafted_10_core", (10, 10, 10)),
    ("linear_20", "linear", "handcrafted_20_core", None),
    ("mlp_20", "mlp", "handcrafted_20_core", (20, 10, 10)),
    ("linear_41", "linear", "handcrafted_41", None),
    ("mlp_41", "mlp", "handcrafted_41", (41, 10, 10)),
)


class UnknownMorpionEvaluatorFamilyPresetError(ValueError):
    """Raised when one Morpion evaluator-family preset name is unknown."""

    def __init__(self, preset_name: str) -> None:
        """Initialize the error with the unknown preset name."""
        super().__init__(f"Unknown Morpion evaluator family preset: {preset_name!r}.")


@dataclass(frozen=True, slots=True)
class _CanonicalFamilySpec:
    """Internal data for one canonical family evaluator spec."""

    name: str
    model_type: str
    feature_subset_name: str
    hidden_sizes: tuple[int, ...] | None

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the built-in subset for this family member."""
        return morpion_feature_subset_from_name(self.feature_subset_name)


def canonical_morpion_evaluator_specs() -> dict[str, MorpionEvaluatorSpec]:
    """Return the canonical eight-evaluator Morpion family specs by name."""
    return {
        spec.name: MorpionEvaluatorSpec(
            name=spec.name,
            model_type=spec.model_type,
            hidden_sizes=spec.hidden_sizes,
            num_epochs=5,
            batch_size=64,
            learning_rate=1e-3,
            feature_subset_name=spec.feature_subset.name,
            feature_names=spec.feature_subset.feature_names,
        )
        for spec in _canonical_family_specs()
    }


def canonical_morpion_evaluator_names() -> tuple[str, ...]:
    """Return canonical evaluator names in stable family order."""
    return tuple(spec.name for spec in _canonical_family_specs())


def canonical_morpion_evaluator_family_config() -> MorpionEvaluatorsConfig:
    """Return the canonical eight-evaluator Morpion bootstrap family config."""
    return MorpionEvaluatorsConfig(evaluators=dict(canonical_morpion_evaluator_specs()))


def entity_token_transformer_small_morpion_evaluator_spec() -> MorpionEvaluatorSpec:
    """Return the small Coral entity-token transformer Morpion evaluator spec."""
    return MorpionEvaluatorSpec(
        name="entity_token_transformer_small",
        model_type=MORPION_ENTITY_TOKEN_MODEL_KIND,
        hidden_sizes=None,
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        entity_max_tokens=1536,
        entity_input_feature_dim=MORPION_ENTITY_TOKEN_FEATURE_DIM,
        entity_d_model=64,
        entity_n_head=4,
        entity_n_layer=2,
        entity_dim_feedforward=256,
        entity_dropout_ratio=0.0,
        entity_pooling="value_token",
        entity_output_tanh=False,
    )


def entity_token_relational_transformer_small_morpion_evaluator_spec() -> (
    MorpionEvaluatorSpec
):
    """Return the canonical small relational entity-token evaluator spec."""
    return MorpionEvaluatorSpec(
        name="entity_token_relational_transformer_small",
        model_type=MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
        hidden_sizes=None,
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        entity_max_tokens=1536,
        entity_input_feature_dim=MORPION_ENTITY_TOKEN_FEATURE_DIM,
        entity_d_model=64,
        entity_n_head=4,
        entity_n_layer=2,
        entity_dim_feedforward=256,
        entity_dropout_ratio=0.0,
        entity_pooling="value_token",
        entity_output_tanh=False,
        entity_use_validity_feature=True,
        entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
        entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
    )


def canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config() -> (
    MorpionEvaluatorsConfig
):
    """Return canonical linear/MLP evaluators plus one Coral entity-token model."""
    evaluators = dict(canonical_morpion_evaluator_specs())
    entity_spec = entity_token_transformer_small_morpion_evaluator_spec()
    evaluators[entity_spec.name] = entity_spec
    return MorpionEvaluatorsConfig(evaluators=evaluators)


def canonical_linear_mlp_entity_and_relational_transformers_small_morpion_evaluator_family_config() -> (
    MorpionEvaluatorsConfig
):
    """Return baselines plus ordinary and relational entity-token models."""
    evaluators = dict(canonical_morpion_evaluator_specs())

    ordinary_spec = entity_token_transformer_small_morpion_evaluator_spec()
    relational_spec = entity_token_relational_transformer_small_morpion_evaluator_spec()

    evaluators[ordinary_spec.name] = ordinary_spec
    evaluators[relational_spec.name] = relational_spec

    return MorpionEvaluatorsConfig(evaluators=evaluators)


def morpion_evaluators_config_from_preset(preset_name: str) -> MorpionEvaluatorsConfig:
    """Resolve a named Morpion evaluator-family preset into explicit specs."""
    if preset_name == CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET:
        return canonical_morpion_evaluator_family_config()
    if (
        preset_name
        == CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
    ):
        return canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config()
    if (
        preset_name
        == CANONICAL_LINEAR_MLP_ENTITY_AND_RELATIONAL_TRANSFORMERS_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
    ):
        return canonical_linear_mlp_entity_and_relational_transformers_small_morpion_evaluator_family_config()
    raise UnknownMorpionEvaluatorFamilyPresetError(preset_name)


def _canonical_family_specs() -> tuple[_CanonicalFamilySpec, ...]:
    """Return the internal canonical family specification table."""
    return tuple(
        _CanonicalFamilySpec(
            name=name,
            model_type=model_type,
            feature_subset_name=feature_subset_name,
            hidden_sizes=hidden_sizes,
        )
        for name, model_type, feature_subset_name, hidden_sizes in _CANONICAL_MORPION_EVALUATOR_SPECS
    )


__all__ = [
    "CANONICAL_LINEAR_MLP_ENTITY_AND_RELATIONAL_TRANSFORMERS_SMALL_MORPION_EVALUATOR_FAMILY_PRESET",
    "CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET",
    "CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET",
    "UnknownMorpionEvaluatorFamilyPresetError",
    "canonical_linear_mlp_entity_and_relational_transformers_small_morpion_evaluator_family_config",
    "canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config",
    "canonical_morpion_evaluator_family_config",
    "canonical_morpion_evaluator_names",
    "canonical_morpion_evaluator_specs",
    "entity_token_relational_transformer_small_morpion_evaluator_spec",
    "entity_token_transformer_small_morpion_evaluator_spec",
    "morpion_evaluators_config_from_preset",
]
