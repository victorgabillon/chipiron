"""Tensor conversion for handcrafted Morpion neural-network features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    extract_morpion_features,
    morpion_feature_names,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

if TYPE_CHECKING:
    from collections import OrderedDict


@dataclass(frozen=True, slots=True)
class MorpionFeatureTensorConverter:
    """Convert Morpion states into ordered handcrafted feature tensors."""

    dynamics: MorpionDynamics

    def feature_names(self) -> tuple[str, ...]:
        """Return the canonical ordered Morpion feature names."""
        return morpion_feature_names()

    def state_to_feature_dict(self, state: MorpionState) -> OrderedDict[str, float]:
        """Return the ordered handcrafted feature dictionary for ``state``."""
        return extract_morpion_features(state=state, dynamics=self.dynamics)

    def state_to_tensor(self, state: MorpionState) -> Tensor:
        """Return a 1D float32 tensor ordered by :meth:`feature_names`."""
        feature_dict = self.state_to_feature_dict(state)
        values = [feature_dict[name] for name in self.feature_names()]
        return torch.tensor(values, dtype=torch.float32)


def morpion_input_dim() -> int:
    """Return the handcrafted Morpion tensor input dimension."""
    return len(morpion_feature_names())


def morpion_state_to_tensor(
    state: MorpionState,
    dynamics: MorpionDynamics | None = None,
) -> Tensor:
    """Convert ``state`` to a 1D float32 handcrafted feature tensor."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    converter = MorpionFeatureTensorConverter(dynamics=dyn)
    return converter.state_to_tensor(state)


__all__ = [
    "MorpionFeatureTensorConverter",
    "morpion_input_dim",
    "morpion_state_to_tensor",
]
