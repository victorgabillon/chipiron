"""Facade for Morpion neural-network input data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MorpionFeatureSubset,
    full_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class MorpionNNInput:
    """Canonical Morpion feature names paired with their tensor values."""

    feature_names: tuple[str, ...]
    tensor: Tensor


def build_morpion_nn_input(
    state: MorpionState,
    dynamics: MorpionDynamics | None = None,
    feature_subset: MorpionFeatureSubset | None = None,
) -> MorpionNNInput:
    """Build the handcrafted Morpion neural-network input bundle."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    converter = MorpionFeatureTensorConverter(
        dynamics=dyn,
        feature_subset=(
            full_morpion_feature_subset() if feature_subset is None else feature_subset
        ),
    )
    feature_names = converter.feature_names()
    tensor = converter.state_to_tensor(state)
    return MorpionNNInput(feature_names=feature_names, tensor=tensor)


__all__ = [
    "MorpionNNInput",
    "build_morpion_nn_input",
]
