"""Tests for Morpion handcrafted feature extraction."""

from __future__ import annotations

from collections import OrderedDict

from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.state import Variant as MorpionVariant

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    extract_morpion_features,
    morpion_feature_names,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState


def make_standard_state() -> MorpionState:
    """Build the standard Chipiron-facing Morpion state."""
    dynamics = MorpionDynamics()
    return dynamics.wrap_atomheart_state(morpion_initial_state())


def test_initial_state_returns_expected_ordered_float_features() -> None:
    """The standard initial state should expose the full stable feature vector."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    features = extract_morpion_features(state=state, dynamics=dynamics)

    assert isinstance(features, OrderedDict)
    assert tuple(features.keys()) == morpion_feature_names()
    assert len(features) == len(morpion_feature_names())
    assert all(isinstance(value, float) for value in features.values())
    assert features["moves"] == 0.0
    assert features["num_points"] == 36.0
    assert features["legal_action_count"] == float(
        len(dynamics.all_legal_actions(state))
    )
    assert (
        sum(features[f"legal_actions_dir_{dir_index}"] for dir_index in range(4))
        == (features["legal_action_count"])
    )


def test_ordering_is_stable_across_repeated_calls() -> None:
    """Feature extraction should be deterministic for repeated calls."""
    dynamics = MorpionDynamics()
    state = make_standard_state()

    first = extract_morpion_features(state=state, dynamics=dynamics)
    second = extract_morpion_features(state=state, dynamics=dynamics)

    assert tuple(first.keys()) == tuple(second.keys()) == morpion_feature_names()
    assert tuple(first.values()) == tuple(second.values())


def test_hand_built_line_state_counts_candidate_and_shape_features() -> None:
    """A small handcrafted line should expose expected geometric opportunities."""
    dynamics = MorpionDynamics()
    state = MorpionState(
        points=frozenset({(0, 0), (1, 0), (2, 0), (3, 0)}),
        used_unit_segments=frozenset(),
        dir_usage_entries=(),
        moves=1,
        variant=MorpionVariant.TOUCHING_5T,
    )

    features = extract_morpion_features(state=state, dynamics=dynamics)

    assert features["moves"] == 1.0
    assert features["num_points"] == 4.0
    assert features["bbox_width"] == 4.0
    assert features["bbox_height"] == 1.0
    assert features["point_density_in_bbox"] == 1.0
    assert features["segments_4_present_1_missing_geometric"] == 2.0
    assert features["segments_4_present_1_missing_legal"] == float(
        len(dynamics.all_legal_actions(state))
    )
    assert features["occupied_connected_components"] == 1.0
    assert features["largest_occupied_component_size"] == 4.0


def test_hand_built_blocked_line_state_counts_usage_features() -> None:
    """Blocked handcrafted lines should feed overlap and dir-usage features."""
    dynamics = MorpionDynamics()
    state = MorpionState(
        points=frozenset({(0, 0), (1, 0), (2, 0), (3, 0)}),
        used_unit_segments=frozenset({((0, 0), (1, 0))}),
        dir_usage_entries=(
            (((0, 0), 0), 1),
            (((1, 0), 0), 2),
            (((2, 0), 0), 3),
            (((2, 0), 1), 1),
        ),
        moves=2,
        variant=MorpionVariant.TOUCHING_5T,
    )

    features = extract_morpion_features(state=state, dynamics=dynamics)

    assert features["num_used_unit_segments"] == 1.0
    assert features["segments_4_present_1_missing_geometric"] == 2.0
    assert features["segments_4_present_1_missing_overlap_ok"] == 0.0
    assert features["segments_4_present_1_missing_legal"] == 0.0
    assert features["dir_usage_value_0_count"] == 12.0
    assert features["dir_usage_value_1_count"] == 2.0
    assert features["dir_usage_value_2_count"] == 1.0
    assert features["dir_usage_value_3_count"] == 1.0
    assert features["points_with_any_dir_usage_3"] == 1.0
    assert features["points_with_ge_2_nonzero_dir_usages"] == 1.0


def test_features_extract_cleanly_after_one_legal_move() -> None:
    """A one-step advanced state should keep the same feature order and progress."""
    dynamics = MorpionDynamics()
    state = make_standard_state()
    first_action = dynamics.legal_actions(state).get_all()[0]
    next_state = dynamics.step(state, first_action).next_state

    features = extract_morpion_features(state=next_state, dynamics=dynamics)

    assert tuple(features.keys()) == morpion_feature_names()
    assert len(features) == len(morpion_feature_names())
    assert all(isinstance(value, float) for value in features.values())
    assert features["moves"] == 1.0
    assert features["num_points"] == 37.0
    assert features["legal_action_count"] == float(
        len(dynamics.all_legal_actions(next_state))
    )
