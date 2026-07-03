"""Tests for canonical Morpion bootstrap evaluator-family presets."""

from __future__ import annotations

import pytest

from chipiron.environments.morpion.bootstrap import (
    CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET,
    CANONICAL_LINEAR_MLP_GRAPH_SMALL_MORPION_EVALUATOR_FAMILY_PRESET,
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    ConflictingMorpionEvaluatorConfigurationError,
    MorpionBootstrapArgs,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnknownMorpionEvaluatorFamilyPresetError,
    canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config,
    canonical_linear_mlp_graph_small_morpion_evaluator_family_config,
    canonical_morpion_evaluator_family_config,
    canonical_morpion_evaluator_names,
    canonical_morpion_evaluator_specs,
    entity_token_transformer_small_morpion_evaluator_spec,
    graph_transformer_small_morpion_evaluator_spec,
    morpion_evaluators_config_from_preset,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    morpion_training_args_from_evaluator_spec,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_ENTITY_TOKEN_TRANSFORMER_MODEL_KIND,
    MORPION_GRAPH_MODEL_KIND,
)


def test_canonical_evaluator_family_contains_exact_expected_members() -> None:
    """The canonical family helper should build the exact 8 expected evaluators."""
    config = canonical_morpion_evaluator_family_config()

    assert set(config.evaluators) == {
        "linear_5",
        "mlp_5",
        "linear_10",
        "mlp_10",
        "linear_20",
        "mlp_20",
        "linear_41",
        "mlp_41",
    }
    assert config.evaluators["linear_5"].model_type == "linear"
    assert config.evaluators["linear_5"].hidden_sizes is None
    assert config.evaluators["linear_5"].feature_subset_name == "handcrafted_5_core"
    assert config.evaluators["mlp_5"].hidden_sizes == (5, 10, 10)
    assert config.evaluators["mlp_10"].hidden_sizes == (10, 10, 10)
    assert config.evaluators["mlp_20"].hidden_sizes == (20, 10, 10)
    assert config.evaluators["mlp_41"].hidden_sizes == (41, 10, 10)
    assert len(config.evaluators["linear_10"].feature_names) == 10
    assert len(config.evaluators["linear_20"].feature_names) == 20
    assert len(config.evaluators["linear_41"].feature_names) == 41
    assert "graph_transformer_small" not in config.evaluators


def test_canonical_evaluator_family_specs_helper_matches_config() -> None:
    """The specs helper and config helper should expose the same evaluator set."""
    specs = canonical_morpion_evaluator_specs()
    config = canonical_morpion_evaluator_family_config()

    assert isinstance(specs, dict)
    assert specs == config.evaluators


def test_canonical_evaluator_name_helper_is_stable() -> None:
    """The canonical evaluator-name helper should return the stable family order."""
    assert canonical_morpion_evaluator_names() == (
        "linear_5",
        "mlp_5",
        "linear_10",
        "mlp_10",
        "linear_20",
        "mlp_20",
        "linear_41",
        "mlp_41",
    )


def test_family_preset_resolution_returns_canonical_family() -> None:
    """The preset resolver should return the canonical family config."""
    resolved = morpion_evaluators_config_from_preset(
        CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    )

    assert resolved == canonical_morpion_evaluator_family_config()


def test_graph_small_family_extends_canonical_without_changing_it() -> None:
    """The graph preset should be opt-in and preserve canonical members."""
    canonical = canonical_morpion_evaluator_family_config()
    graph_family = canonical_linear_mlp_graph_small_morpion_evaluator_family_config()

    assert set(canonical.evaluators) == set(canonical_morpion_evaluator_names())
    assert "graph_transformer_small" not in canonical.evaluators
    assert set(graph_family.evaluators) == {
        *canonical_morpion_evaluator_names(),
        "graph_transformer_small",
    }
    for name in canonical_morpion_evaluator_names():
        assert graph_family.evaluators[name] == canonical.evaluators[name]


def test_graph_small_family_preset_resolves() -> None:
    """The preset resolver should expose the opt-in graph evaluator family."""
    resolved = morpion_evaluators_config_from_preset(
        CANONICAL_LINEAR_MLP_GRAPH_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
    )

    assert (
        resolved == canonical_linear_mlp_graph_small_morpion_evaluator_family_config()
    )
    graph_spec = resolved.evaluators["graph_transformer_small"]
    assert graph_spec.name == "graph_transformer_small"
    assert graph_spec.model_type == MORPION_GRAPH_MODEL_KIND


def test_entity_transformer_small_family_extends_canonical_without_changing_it() -> (
    None
):
    """The Coral entity-token preset should preserve canonical linear/MLP members."""
    canonical = canonical_morpion_evaluator_family_config()
    entity_family = (
        canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config()
    )

    assert set(canonical.evaluators) == set(canonical_morpion_evaluator_names())
    assert "entity_token_transformer_small" not in canonical.evaluators
    assert set(entity_family.evaluators) == {
        *canonical_morpion_evaluator_names(),
        "entity_token_transformer_small",
    }
    for name in canonical_morpion_evaluator_names():
        assert entity_family.evaluators[name] == canonical.evaluators[name]


def test_entity_transformer_small_family_preset_resolves() -> None:
    """The preset resolver should expose the Coral entity-token evaluator family."""
    resolved = morpion_evaluators_config_from_preset(
        CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
    )

    assert (
        resolved
        == canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config()
    )
    entity_spec = resolved.evaluators["entity_token_transformer_small"]
    assert entity_spec.name == "entity_token_transformer_small"
    assert entity_spec.model_type == MORPION_ENTITY_TOKEN_TRANSFORMER_MODEL_KIND


def test_graph_transformer_small_spec_uses_laptop_safe_defaults() -> None:
    """The catalogue graph evaluator should be small enough for opt-in smoke runs."""
    spec = graph_transformer_small_morpion_evaluator_spec()

    assert spec.name == "graph_transformer_small"
    assert spec.model_type == MORPION_GRAPH_MODEL_KIND
    assert spec.batch_size <= 16
    assert spec.graph_max_tokens == 1536
    assert spec.graph_d_model == 64
    assert spec.graph_n_head == 4
    assert spec.graph_n_layer == 2
    assert spec.graph_dim_feedforward == 256
    assert spec.graph_dropout_ratio == 0.0
    assert spec.graph_pooling == "value_token"
    assert spec.graph_output_tanh is False


def test_entity_token_transformer_small_spec_uses_laptop_safe_defaults() -> None:
    """The Coral entity-token evaluator should use the safe small defaults."""
    spec = entity_token_transformer_small_morpion_evaluator_spec()

    assert spec.name == "entity_token_transformer_small"
    assert spec.model_type == MORPION_ENTITY_TOKEN_TRANSFORMER_MODEL_KIND
    assert spec.batch_size <= 16
    assert spec.graph_max_tokens == 1536
    assert spec.graph_d_model == 64
    assert spec.graph_n_head == 4
    assert spec.graph_n_layer == 2
    assert spec.graph_dim_feedforward == 256
    assert spec.graph_dropout_ratio == 0.0
    assert spec.graph_pooling == "value_token"
    assert spec.graph_output_tanh is False


def test_training_args_from_graph_spec_preserves_graph_fields() -> None:
    """cycle_training should pass graph catalogue fields into training args."""
    spec = MorpionEvaluatorSpec(
        name="graph_custom",
        model_type=MORPION_GRAPH_MODEL_KIND,
        hidden_sizes=None,
        num_epochs=7,
        batch_size=3,
        learning_rate=2e-3,
        graph_max_tokens=321,
        graph_d_model=32,
        graph_n_head=4,
        graph_n_layer=1,
        graph_dim_feedforward=64,
        graph_dropout_ratio=0.1,
        graph_pooling="masked_mean",
        graph_output_tanh=False,
    )

    training_args = morpion_training_args_from_evaluator_spec(
        spec=spec,
        dataset_file="/tmp/morpion_rows.json",
        output_dir="/tmp/morpion_model",
        shuffle=False,
        validation_fraction=0.125,
        validation_seed=17,
        device="cpu",
    )

    assert training_args.model_kind == MORPION_GRAPH_MODEL_KIND
    assert training_args.num_epochs == 7
    assert training_args.batch_size == 3
    assert training_args.learning_rate == 2e-3
    assert training_args.graph_max_tokens == 321
    assert training_args.graph_d_model == 32
    assert training_args.graph_n_head == 4
    assert training_args.graph_n_layer == 1
    assert training_args.graph_dim_feedforward == 64
    assert training_args.graph_dropout_ratio == 0.1
    assert training_args.graph_pooling == "masked_mean"
    assert training_args.graph_output_tanh is False
    assert training_args.validation_fraction == 0.125
    assert training_args.validation_seed == 17


def test_training_args_from_entity_transformer_spec_uses_coral_model_kind() -> None:
    """cycle_training should pass the Coral model kind through unchanged."""
    spec = entity_token_transformer_small_morpion_evaluator_spec()

    training_args = morpion_training_args_from_evaluator_spec(
        spec=spec,
        dataset_file="/tmp/morpion_rows.json",
        output_dir="/tmp/morpion_model",
        shuffle=False,
        validation_fraction=0.125,
        validation_seed=17,
        device="cpu",
    )

    assert training_args.model_kind == MORPION_ENTITY_TOKEN_TRANSFORMER_MODEL_KIND
    assert training_args.graph_input_feature_dim == spec.graph_input_feature_dim
    assert training_args.graph_pooling == "value_token"


def test_unknown_family_preset_fails_clearly() -> None:
    """Unknown family presets should raise a dedicated error."""
    with pytest.raises(UnknownMorpionEvaluatorFamilyPresetError):
        morpion_evaluators_config_from_preset("missing_preset")


def test_bootstrap_args_can_resolve_canonical_family_preset() -> None:
    """Bootstrap args should resolve the family preset into explicit evaluators."""
    args = MorpionBootstrapArgs(
        work_dir="/tmp/morpion-family",
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )

    assert (
        args.resolved_evaluators_config() == canonical_morpion_evaluator_family_config()
    )


def test_bootstrap_args_can_resolve_graph_family_preset() -> None:
    """Bootstrap args should resolve the opt-in graph evaluator family preset."""
    args = MorpionBootstrapArgs(
        work_dir="/tmp/morpion-family",
        evaluator_family_preset=(
            CANONICAL_LINEAR_MLP_GRAPH_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
        ),
    )

    assert (
        args.resolved_evaluators_config()
        == canonical_linear_mlp_graph_small_morpion_evaluator_family_config()
    )


def test_bootstrap_args_can_resolve_entity_transformer_family_preset() -> None:
    """Bootstrap args should resolve the Coral entity-token family preset."""
    args = MorpionBootstrapArgs(
        work_dir="/tmp/morpion-family",
        evaluator_family_preset=(
            CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
        ),
    )

    assert (
        args.resolved_evaluators_config()
        == canonical_linear_mlp_entity_transformer_small_morpion_evaluator_family_config()
    )


def test_bootstrap_args_reject_preset_and_explicit_config_together() -> None:
    """The preset path and explicit evaluator config should be mutually exclusive."""
    explicit = MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )
        }
    )
    args = MorpionBootstrapArgs(
        work_dir="/tmp/morpion-family",
        evaluators_config=explicit,
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )

    with pytest.raises(ConflictingMorpionEvaluatorConfigurationError):
        args.resolved_evaluators_config()
