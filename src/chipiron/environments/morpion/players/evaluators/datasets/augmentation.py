"""On-demand state augmentation without duplicating stored training rows."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

import torch
from torch.utils.data import Dataset

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.symmetry import transform_morpion_state
from chipiron.learning.supervised import TensorSupervisedBatch

if TYPE_CHECKING:
    from chipiron.environments.morpion.types import MorpionState


class CanonicalStates(Protocol):
    """A memory or disk backed sequence of original state/target pairs."""

    def __len__(self) -> int:
        """Return the original source row count."""
        ...

    def __getitem__(self, index: int, /) -> tuple[MorpionState, float]:
        """Return one canonical state and its unchanged target."""
        ...


def training_symmetry(*, seed: int, epoch: int, row_index: int, enabled: bool) -> int:
    """Choose a reproducible uniform D4 element independent of worker/order RNG.

    Epoch -1 denotes canonical validation. Separate hashing avoids consuming
    model initialization, dropout, or shuffle random numbers in the candidate.
    """
    if not enabled or epoch < 0:
        return 0
    key = f"morpion_d4_v1:{seed}:{epoch}:{row_index}".encode()
    return hashlib.blake2b(key, digest_size=1).digest()[0] % 8


class AugmentedMorpionDataset(Dataset[TensorSupervisedBatch]):
    """Convert a (row index, epoch) request after optional geometric augmentation."""

    def __init__(
        self,
        rows: CanonicalStates,
        *,
        seed: int,
        augment: bool,
        max_tokens: int = 1536,
    ) -> None:
        """Keep one copy of the original rows and no eightfold tensor cache."""
        self.rows = rows
        self.seed = seed
        self.augment = augment
        self.converter = MorpionRelationalEntityTokenConverter(max_tokens=max_tokens)

    def __len__(self) -> int:
        """Return the original row count, regardless of augmentation."""
        return len(self.rows)

    def __getitem__(self, key: tuple[int, int]) -> TensorSupervisedBatch:
        """Return original target and tokens rebuilt from the selected symmetry."""
        index, epoch = key
        state, target = self.rows[index]
        symmetry = training_symmetry(
            seed=self.seed, epoch=epoch, row_index=index, enabled=self.augment
        )
        tensors = self.converter.state_to_tensors(
            transform_morpion_state(state, symmetry)
        )
        return TensorSupervisedBatch(
            input_tensor=tensors.token_tensor,
            auxiliary_input_tensors=(tensors.relation_triples,),
            target_tensor=torch.tensor([target], dtype=torch.float32),
            is_batch=False,
        )
