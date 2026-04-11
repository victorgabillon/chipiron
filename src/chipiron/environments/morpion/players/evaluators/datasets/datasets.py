"""PyTorch dataset helpers for Morpion supervised learning rows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    decode_morpion_state_ref_payload,
    load_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionFeatureTensorConverter,
    morpion_feature_names,
)
from chipiron.environments.morpion.types import MorpionDynamics

_FEATURE_NAMES: tuple[str, ...] = tuple(str(name) for name in morpion_feature_names())


@dataclass(frozen=True, slots=True)
class MorpionSupervisedDatasetArgs:
    """Arguments for loading one persisted Morpion supervised-row artifact."""

    file_name: str | os.PathLike[str]


class MorpionSupervisedSample(NamedTuple):
    """Tuple-like Morpion sample with small trainer-compatibility helpers."""

    input_tensor: Tensor
    target_tensor: Tensor

    @property
    def is_batch(self) -> bool:
        """Return whether this sample contains batched tensors."""
        return cast("bool", self.input_tensor.ndim > 1)

    def get_input_layer(self) -> Tensor:
        """Return the feature tensor for the sample."""
        return self.input_tensor

    def get_target_value(self) -> Tensor:
        """Return the regression target tensor for the sample."""
        return self.target_tensor


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
        self._converter = MorpionFeatureTensorConverter(dynamics=self._dynamics)
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
        return len(_FEATURE_NAMES)

    def feature_names(self) -> tuple[str, ...]:
        """Return the canonical handcrafted Morpion feature ordering."""
        return _FEATURE_NAMES


def load_morpion_supervised_dataset(
    args: MorpionSupervisedDatasetArgs,
) -> MorpionSupervisedDataset:
    """Load one eager Morpion supervised dataset from persisted raw rows."""
    return MorpionSupervisedDataset(args)


__all__ = [
    "MorpionSupervisedDataset",
    "MorpionSupervisedDatasetArgs",
    "load_morpion_supervised_dataset",
    "process_morpion_supervised_row_to_tensors",
]
