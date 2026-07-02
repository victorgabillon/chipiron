"""Tests for the Morpion neural-network training package API."""

from __future__ import annotations


def test_training_package_exports_public_api() -> None:
    """The training package should expose the public trainer API."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionStreamingTrainingArgs,
        MorpionTrainingArgs,
        train_morpion_regressor,
        train_morpion_regressor_streaming,
    )

    assert MorpionTrainingArgs is not None
    assert MorpionStreamingTrainingArgs is not None
    assert callable(train_morpion_regressor)
    assert callable(train_morpion_regressor_streaming)


def test_old_train_module_is_thin_reexport() -> None:
    """The old train module should still expose the trainer entry points."""
    from chipiron.environments.morpion.players.evaluators.neural_networks import train

    assert train.train_morpion_regressor is not None
    assert train.train_morpion_regressor_streaming is not None
