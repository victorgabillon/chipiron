"""Morpion neural-network training argument dataclasses."""

from __future__ import annotations

import math
import os  # noqa: TC003 - keep available for typing.get_type_hints.
from collections.abc import Callable
from dataclasses import dataclass, field

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
)

type MorpionTrainingProgressCallback = Callable[[int, int, int, int], None]


class InvalidValidationFractionError(ValueError):
    """Raised when supervised-training validation splitting is outside bounds."""

    def __init__(self) -> None:
        """Initialize the invalid-validation-fraction error."""
        super().__init__("validation_fraction must be in [0.0, 1.0).")


@dataclass(frozen=True, slots=True)
class MorpionTrainingArgs:
    """Arguments for the Morpion supervised-regression training helper."""

    dataset_file: str | os.PathLike[str]
    output_dir: str | os.PathLike[str]
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_sizes: tuple[int, ...] | None = None
    hidden_dim: int | None = None
    graph_max_tokens: int = 1536
    graph_input_feature_dim: int = MORPION_GRAPH_TOKEN_FEATURE_DIM
    graph_d_model: int = 64
    graph_n_head: int = 4
    graph_n_layer: int = 2
    graph_dim_feedforward: int = 256
    graph_dropout_ratio: float = 0.0
    graph_pooling: str = "value_token"
    graph_output_tanh: bool = False
    validation_fraction: float = 0.2
    validation_seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if (
            not math.isfinite(self.validation_fraction)
            or self.validation_fraction < 0.0
            or self.validation_fraction >= 1.0
        ):
            raise InvalidValidationFractionError
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this training job."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionStreamingTrainingArgs:
    """Arguments for streaming Morpion supervised-regression training."""

    training_args: MorpionTrainingArgs
    row_chunk_size: int = 8192
    max_rows: int | None = None
    progress_callback: MorpionTrainingProgressCallback | None = None

    def __post_init__(self) -> None:
        """Validate streaming controls."""
        if isinstance(self.row_chunk_size, bool) or self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be a positive integer.")  # noqa: TRY003
        if self.max_rows is not None and (
            isinstance(self.max_rows, bool) or self.max_rows < 0
        ):
            raise ValueError("max_rows must be a non-negative integer or None.")  # noqa: TRY003
