"""Simple Morpion regressors over handcrafted feature tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from coral.chi_nn import ChiNN
from torch import Tensor, nn

from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    morpion_input_dim,
)

MORPION_FEATURE_SCHEMA = "morpion_handcrafted_v1"
MORPION_INPUT_DIM = morpion_input_dim()


@dataclass(frozen=True, slots=True)
class MorpionRegressorArgs:
    """Arguments for the Morpion handcrafted-feature regressor."""

    model_kind: str = "linear"
    input_dim: int = field(default=MORPION_INPUT_DIM)
    hidden_sizes: tuple[int, ...] | None = None


class UnsupportedMorpionModelKindError(ValueError):
    """Raised when a Morpion regressor kind is unsupported."""

    def __init__(self, model_kind: str) -> None:
        """Initialize the error with the unsupported model kind."""
        super().__init__(f"Unsupported Morpion model_kind: {model_kind!r}.")


class MissingMorpionHiddenSizesError(ValueError):
    """Raised when an MLP regressor is missing its hidden sizes."""

    def __init__(self) -> None:
        """Initialize the error for missing hidden sizes."""
        super().__init__("`hidden_sizes` is required when model_kind='mlp'.")


class MissingMorpionHiddenDimError(MissingMorpionHiddenSizesError):
    """Backward-compatible alias for older callers expecting the old error name."""


def _build_model_module(args: MorpionRegressorArgs) -> nn.Module:
    """Build the internal torch module for one Morpion regressor."""
    if args.model_kind == "linear":
        return nn.Linear(args.input_dim, 1)
    if args.model_kind == "mlp":
        if not args.hidden_sizes:
            raise MissingMorpionHiddenSizesError

        layers: list[nn.Module] = []
        previous_dim = args.input_dim
        for hidden_size in args.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_size),
                    nn.ReLU(),
                ]
            )
            previous_dim = hidden_size
        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers)
    raise UnsupportedMorpionModelKindError(args.model_kind)


class MorpionRegressor(ChiNN):
    """Tiny Morpion value regressor over handcrafted feature tensors."""

    args: MorpionRegressorArgs
    net: nn.Module

    def __init__(self, args: MorpionRegressorArgs) -> None:
        """Build a Morpion regressor from explicit args."""
        super().__init__()
        self.args = args
        self.net = _build_model_module(args)

    def forward(self, x: Tensor) -> Tensor:
        """Run the regressor on one unbatched or batched feature tensor."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return cast("Tensor", self.net(x))

    def init_weights(self) -> None:
        """Keep PyTorch's default initialization for the first Morpion model."""
        return

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """Write the current state dict to a readable JSON file."""
        readable_state = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self.state_dict().items()
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(readable_state, handle, indent=2, sort_keys=True)


def build_morpion_regressor(
    args: MorpionRegressorArgs | None = None,
) -> MorpionRegressor:
    """Build one Morpion regressor with default or explicit args."""
    resolved_args = args if args is not None else MorpionRegressorArgs()
    return MorpionRegressor(resolved_args)


__all__ = [
    "MORPION_FEATURE_SCHEMA",
    "MORPION_INPUT_DIM",
    "MissingMorpionHiddenDimError",
    "MissingMorpionHiddenSizesError",
    "MorpionRegressor",
    "MorpionRegressorArgs",
    "UnsupportedMorpionModelKindError",
    "build_morpion_regressor",
]
