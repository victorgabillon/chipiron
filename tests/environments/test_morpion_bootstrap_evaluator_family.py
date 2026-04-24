"""Tests for canonical Morpion bootstrap evaluator-family presets."""

from __future__ import annotations

import pytest

from chipiron.environments.morpion.bootstrap import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    ConflictingMorpionEvaluatorConfigurationError,
    MorpionBootstrapArgs,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnknownMorpionEvaluatorFamilyPresetError,
    canonical_morpion_evaluator_family_config,
    canonical_morpion_evaluator_names,
    canonical_morpion_evaluator_specs,
    morpion_evaluators_config_from_preset,
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
