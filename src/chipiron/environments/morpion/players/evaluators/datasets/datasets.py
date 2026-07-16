"""PyTorch dataset helpers for Morpion supervised learning rows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    decode_morpion_state_ref_payload,
    load_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class MorpionSupervisedDatasetArgs:
    """Arguments for loading one persisted Morpion supervised-row artifact."""

    file_name: str | os.PathLike[str]
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this dataset."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


MorpionSupervisedSample = TensorSupervisedBatch
MorpionEntityTokenSupervisedSample = TensorSupervisedBatch


@dataclass(frozen=True, slots=True)
class MorpionEntityTokenSupervisedDatasetArgs:
    """Arguments for loading Morpion rows as entity-token samples."""

    file_name: str | os.PathLike[str]
    max_tokens: int = 1536


def process_morpion_supervised_row_to_tensors(
    row: MorpionSupervisedRow,
    *,
    dynamics: MorpionDynamics | None = None,
    converter: MorpionFeatureTensorConverter | None = None,
) -> MorpionSupervisedSample:
    """Convert one raw Morpion supervised row into input and target tensors."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    feature_converter = (
        converter
        if converter is not None
        else MorpionFeatureTensorConverter(dynamics=dyn)
    )
    atom_state = decode_morpion_state_ref_payload(row.state_ref_payload)
    chipiron_state = dyn.wrap_atomheart_state(atom_state)
    input_tensor = feature_converter.state_to_tensor(chipiron_state)
    target_tensor = torch.tensor([row.target_value], dtype=torch.float32)
    return MorpionSupervisedSample(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        is_batch=False,
    )


def process_morpion_supervised_row_to_entity_token_tensors(
    row: MorpionSupervisedRow,
    *,
    dynamics: MorpionDynamics | None = None,
    converter: MorpionEntityTokenConverter | None = None,
) -> MorpionEntityTokenSupervisedSample:
    """Convert one raw Morpion supervised row into entity-token tensors."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    entity_converter = (
        converter
        if converter is not None
        else MorpionEntityTokenConverter(dynamics=dyn)
    )
    atom_state = decode_morpion_state_ref_payload(row.state_ref_payload)
    chipiron_state = dyn.wrap_atomheart_state(atom_state)
    input_tensor = entity_converter.state_to_tensor(chipiron_state)
    target_tensor = torch.tensor([row.target_value], dtype=torch.float32)
    return MorpionEntityTokenSupervisedSample(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        is_batch=False,
    )


class MorpionSupervisedDataset(Dataset[MorpionSupervisedSample]):
    """Eager in-memory dataset for Morpion supervised regression rows."""

    args: MorpionSupervisedDatasetArgs
    _dynamics: MorpionDynamics
    _converter: MorpionFeatureTensorConverter
    _rows_bundle: MorpionSupervisedRows
    _samples: tuple[MorpionSupervisedSample, ...]

    def __init__(self, args: MorpionSupervisedDatasetArgs) -> None:
        """Load and eagerly preprocess one persisted Morpion row file."""
        self.args = args
        self._dynamics = MorpionDynamics()
        self._converter = MorpionFeatureTensorConverter(
            dynamics=self._dynamics,
            feature_subset=args.feature_subset,
        )
        self._rows_bundle = load_morpion_supervised_rows(os.fspath(args.file_name))
        self._samples = tuple(
            process_morpion_supervised_row_to_tensors(
                row,
                dynamics=self._dynamics,
                converter=self._converter,
            )
            for row in self._rows_bundle.rows
        )

    def __len__(self) -> int:
        """Return the number of eagerly preprocessed Morpion samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> MorpionSupervisedSample:
        """Return one preprocessed Morpion supervised sample."""
        return self._samples[index]

    @property
    def input_dim(self) -> int:
        """Return the handcrafted Morpion feature dimension."""
        return self._converter.input_dim

    def feature_names(self) -> tuple[str, ...]:
        """Return the canonical handcrafted Morpion feature ordering."""
        return self._converter.feature_names()


class MorpionEntityTokenSupervisedDataset(Dataset[MorpionEntityTokenSupervisedSample]):
    """Eager in-memory dataset for entity-token Morpion regression rows."""

    args: MorpionEntityTokenSupervisedDatasetArgs
    _dynamics: MorpionDynamics
    _converter: MorpionEntityTokenConverter
    _rows_bundle: MorpionSupervisedRows
    _samples: tuple[MorpionEntityTokenSupervisedSample, ...]

    def __init__(self, args: MorpionEntityTokenSupervisedDatasetArgs) -> None:
        """Load and eagerly preprocess one persisted Morpion row file."""
        self.args = args
        self._dynamics = MorpionDynamics()
        self._converter = MorpionEntityTokenConverter(
            dynamics=self._dynamics,
            max_tokens=args.max_tokens,
        )
        self._rows_bundle = load_morpion_supervised_rows(os.fspath(args.file_name))
        self._samples = tuple(
            process_morpion_supervised_row_to_entity_token_tensors(
                row,
                dynamics=self._dynamics,
                converter=self._converter,
            )
            for row in self._rows_bundle.rows
        )

    def __len__(self) -> int:
        """Return the number of eagerly preprocessed Morpion samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> MorpionEntityTokenSupervisedSample:
        """Return one preprocessed entity-token Morpion sample."""
        return self._samples[index]

    @property
    def input_dim(self) -> int:
        """Return the entity-token feature dimension."""
        return self._converter.input_dim

    def feature_names(self) -> tuple[str, ...]:
        """Return the entity-token feature ordering."""
        return self._converter.feature_names()


def collate_morpion_entity_token_supervised_samples(
    samples: Sequence[MorpionEntityTokenSupervisedSample],
) -> MorpionEntityTokenSupervisedSample:
    """Pad variable-length entity-token samples into one batch."""
    if not samples:
        return MorpionEntityTokenSupervisedSample(
            input_tensor=torch.empty((0, 0, 0), dtype=torch.float32),
            target_tensor=torch.empty((0, 1), dtype=torch.float32),
            is_batch=True,
        )
    batch_size = len(samples)
    max_token_count = max(sample.input_tensor.shape[0] for sample in samples)
    feature_dim = samples[0].input_tensor.shape[1]
    input_tensor = torch.zeros(
        (batch_size, max_token_count, feature_dim),
        dtype=samples[0].input_tensor.dtype,
    )
    target_tensor = torch.zeros((batch_size, 1), dtype=samples[0].target_tensor.dtype)
    for index, sample in enumerate(samples):
        token_count = sample.input_tensor.shape[0]
        input_tensor[index, :token_count, :] = sample.input_tensor
        target_tensor[index, :] = sample.target_tensor.reshape(1)
    return MorpionEntityTokenSupervisedSample(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        is_batch=True,
    )


def collate_morpion_supervised_samples(
    samples: Sequence[MorpionSupervisedSample],
) -> MorpionSupervisedSample:
    """Stack fixed-width Morpion samples into one supervised tensor batch."""
    if not samples:
        return MorpionSupervisedSample(
            input_tensor=torch.empty((0, 0), dtype=torch.float32),
            target_tensor=torch.empty((0, 1), dtype=torch.float32),
            is_batch=True,
        )
    return MorpionSupervisedSample(
        input_tensor=torch.stack([sample.input_tensor for sample in samples]),
        target_tensor=torch.stack([sample.target_tensor for sample in samples]),
        is_batch=True,
    )


def load_morpion_supervised_dataset(
    args: MorpionSupervisedDatasetArgs,
) -> MorpionSupervisedDataset:
    """Load one eager Morpion supervised dataset from persisted raw rows."""
    return MorpionSupervisedDataset(args)


__all__ = [
    "MorpionEntityTokenSupervisedDataset",
    "MorpionEntityTokenSupervisedDatasetArgs",
    "MorpionEntityTokenSupervisedSample",
    "MorpionSupervisedDataset",
    "MorpionSupervisedDatasetArgs",
    "collate_morpion_entity_token_supervised_samples",
    "collate_morpion_supervised_samples",
    "load_morpion_supervised_dataset",
    "process_morpion_supervised_row_to_entity_token_tensors",
    "process_morpion_supervised_row_to_tensors",
]
