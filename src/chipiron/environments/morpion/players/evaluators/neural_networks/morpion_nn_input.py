"""Facade for Morpion neural-network input data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    morpion_feature_names,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    morpion_state_to_tensor,
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
) -> MorpionNNInput:
    """Build the handcrafted Morpion neural-network input bundle."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    feature_names = morpion_feature_names()
    tensor = morpion_state_to_tensor(state=state, dynamics=dyn)
    return MorpionNNInput(feature_names=feature_names, tensor=tensor)


__all__ = [
    "MorpionNNInput",
    "build_morpion_nn_input",
]
