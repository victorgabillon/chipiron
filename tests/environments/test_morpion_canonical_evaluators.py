"""Canonical value families retain their architecture through bootstrap configuration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    morpion_training_args_from_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    canonical_morpion_evaluator_specs,
    morpion_evaluators_config_from_preset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    build_morpion_regressor,
    morpion_evaluator_v1_model_args,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.model_args import (
    morpion_regressor_args_from_training_args,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "preset", ["canonical_value_v1", "canonical_8_linear_mlp_transformer_v1"]
)
def test_canonical_config_survives_bootstrap_training_path(
    tmp_path: Path, preset: str
) -> None:
    """Persisted specs reach the model builder with the frozen scale and topology."""
    config = bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=tmp_path, evaluator_family_preset=preset)
    )
    restored = bootstrap_config_from_dict(bootstrap_config_to_dict(config))
    assert restored == config
    specs = restored.evaluators.evaluators
    assert len(specs) == (3 if preset == "canonical_value_v1" else 9)
    for name, expected_count in (
        ("linear_41", 42),
        ("mlp_41", 2263),
        ("transformer_v1", 106049),
    ):
        training = morpion_training_args_from_evaluator_spec(
            spec=specs[name],
            dataset_file=tmp_path / "rows",
            output_dir=tmp_path / name,
            shuffle=True,
            validation_fraction=0.2,
            validation_seed=0,
            device="cpu",
        )
        args = morpion_regressor_args_from_training_args(training)
        assert (
            sum(p.numel() for p in build_morpion_regressor(args).parameters())
            == expected_count
        )
        assert training.num_epochs == 5 and training.learning_rate == 0.001
        if name == "transformer_v1":
            assert args == morpion_evaluator_v1_model_args()
            assert training.batch_size == 8
        else:
            assert specs[name] == canonical_morpion_evaluator_specs()[name]
            assert training.batch_size == 64
            assert not args.target_transform_enabled


def test_legacy_catalog_and_serialization_remain_unchanged(tmp_path: Path) -> None:
    """Default relation scale stays implicit and old eight baseline specs stay intact."""
    old = morpion_evaluators_config_from_preset("canonical_8_linear_mlp_subsets")
    new = morpion_evaluators_config_from_preset("canonical_8_linear_mlp_transformer_v1")
    assert all(new.evaluators[name] == spec for name, spec in old.evaluators.items())
    config = bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=tmp_path, evaluators_config=old)
    )
    payload = bootstrap_config_to_dict(config)
    assert "relation_bias_scale" not in str(payload)
    assert bootstrap_config_from_dict(payload) == config


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -0.1])
def test_invalid_relation_scale_rejected(scale: float) -> None:
    """A malformed canonical spec cannot enter persisted bootstrap configuration."""
    spec = morpion_evaluators_config_from_preset("canonical_value_v1").evaluators[
        "transformer_v1"
    ]
    with pytest.raises(ValueError):
        replace(spec, relation_bias_scale=scale)
