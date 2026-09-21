"""Tests for Morpion handcrafted feature tensor conversion."""

from __future__ import annotations

import torch
from atomheart.games.morpion import initial_state as morpion_initial_state
from torch import Tensor

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    extract_morpion_features,
    morpion_feature_names,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
    morpion_feature_subset_from_feature_names,
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


def test_initial_state_converts_to_float32_1d_tensor() -> None:
    """The standard initial state should convert to a stable float32 vector."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.ndim == 1
    assert tensor.shape == (len(morpion_feature_names()),)
    assert tensor.shape == (morpion_input_dim(),)


def test_converter_and_convenience_function_agree() -> None:
    """The class API and convenience function should produce identical tensors."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    converter = MorpionFeatureTensorConverter(dynamics=dynamics)

    converter_tensor = converter.state_to_tensor(state)
    convenience_tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)

    torch.testing.assert_close(converter_tensor, convenience_tensor)


def test_tensor_values_match_ordered_feature_dict_exactly() -> None:
    """Tensor values should follow the canonical feature-dict order exactly."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    feature_dict = extract_morpion_features(state=state, dynamics=dynamics)
    tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)
    expected = torch.tensor(
        [feature_dict[name] for name in morpion_feature_names()],
        dtype=torch.float32,
    )

    assert tuple(feature_dict.keys()) == morpion_feature_names()
    assert len(feature_dict) == morpion_input_dim()
    assert tensor.shape[0] == len(feature_dict)
    torch.testing.assert_close(tensor, expected)


def test_repeated_calls_are_deterministic() -> None:
    """Repeated tensor conversion calls should be elementwise stable."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    first = morpion_state_to_tensor(state=state, dynamics=dynamics)
    second = morpion_state_to_tensor(state=state, dynamics=dynamics)

    torch.testing.assert_close(first, second)


def test_after_one_legal_move_shape_is_unchanged() -> None:
    """A one-step advanced state should keep the same tensor shape."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    first_action = dynamics.legal_actions(state).get_all()[0]
    next_state = dynamics.step(state, first_action).next_state

    initial_tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)
    moved_tensor = morpion_state_to_tensor(state=next_state, dynamics=dynamics)

    assert initial_tensor.ndim == 1
    assert moved_tensor.ndim == 1
    assert initial_tensor.shape == (morpion_input_dim(),)
    assert moved_tensor.shape == (morpion_input_dim(),)


def test_converter_feature_names_api_is_stable() -> None:
    """The converter should expose the extractor's canonical feature ordering."""
    dynamics = MorpionDynamics()
    converter = MorpionFeatureTensorConverter(dynamics=dynamics)

    assert converter.feature_names() == morpion_feature_names()
    assert len(converter.feature_names()) == morpion_input_dim()


def test_reduced_subset_projects_exact_entries_from_full_tensor() -> None:
    """A reduced subset tensor should match the corresponding full-tensor entries."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    subset = morpion_feature_subset_from_feature_names(
        "handcrafted_5_custom",
        MORPION_CANONICAL_FEATURE_NAMES[:5],
    )

    full_tensor = morpion_state_to_tensor(state=state, dynamics=dynamics)
    subset_tensor = morpion_state_to_tensor(
        state=state,
        dynamics=dynamics,
        feature_subset=subset,
    )

    assert subset_tensor.shape == (subset.dimension,)
    assert morpion_input_dim(subset) == subset.dimension
    torch.testing.assert_close(subset_tensor, full_tensor[: subset.dimension])


def test_converter_reports_subset_feature_names_and_dimension() -> None:
    """The converter metadata should reflect the selected subset exactly."""
    dynamics = MorpionDynamics()
    subset = morpion_feature_subset_from_feature_names(
        "handcrafted_10_custom",
        MORPION_CANONICAL_FEATURE_NAMES[:10],
    )
    converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=subset,
    )

    assert converter.feature_names() == subset.feature_names
    assert converter.input_dim == subset.dimension
