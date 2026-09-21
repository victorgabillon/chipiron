"""Tests for the Morpion neural-network input facade."""

from __future__ import annotations

import torch
from atomheart.games.morpion import initial_state as morpion_initial_state

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    morpion_feature_names,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.morpion_nn_input import (
    MorpionNNInput,
    build_morpion_nn_input,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
    morpion_input_dim,
    morpion_state_to_tensor,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState


def make_standard_state() -> MorpionState:
    """Build the standard Chipiron-facing Morpion state."""
    dynamics = MorpionDynamics()
    return dynamics.wrap_atomheart_state(morpion_initial_state())


def test_build_initial_state_nn_input() -> None:
    """The facade should build names and tensor for the initial state."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    nn_input = build_morpion_nn_input(state=state, dynamics=dynamics)

    assert isinstance(nn_input, MorpionNNInput)
    assert nn_input.feature_names == morpion_feature_names()
    assert nn_input.tensor.ndim == 1
    assert len(nn_input.feature_names) == morpion_input_dim()
    assert nn_input.tensor.shape == (morpion_input_dim(),)


def test_builder_tensor_matches_direct_tensor_function() -> None:
    """The facade tensor should match the direct tensor helper."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    nn_input = build_morpion_nn_input(state=state, dynamics=dynamics)
    direct_tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)

    torch.testing.assert_close(nn_input.tensor, direct_tensor)


def test_builder_names_match_converter_class_names() -> None:
    """The facade should expose the same names as the converter class."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    converter = MorpionFeatureTensorConverter(dynamics=dynamics)

    nn_input = build_morpion_nn_input(state=state, dynamics=dynamics)

    assert nn_input.feature_names == converter.feature_names()


def test_one_step_advanced_state_still_builds_nn_input() -> None:
    """A one-step advanced Morpion state should still build cleanly."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    first_action = dynamics.legal_actions(state).get_all()[0]
    next_state = dynamics.step(state, first_action).next_state

    nn_input = build_morpion_nn_input(state=next_state, dynamics=dynamics)

    assert nn_input.tensor.ndim == 1
    assert len(nn_input.feature_names) == nn_input.tensor.shape[0]
    assert nn_input.feature_names == morpion_feature_names()


def test_repeated_calls_are_deterministic() -> None:
    """Repeated facade builds should preserve names and tensor values."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    first = build_morpion_nn_input(state=state, dynamics=dynamics)
    second = build_morpion_nn_input(state=state, dynamics=dynamics)

    assert first.feature_names == second.feature_names
    torch.testing.assert_close(first.tensor, second.tensor)
