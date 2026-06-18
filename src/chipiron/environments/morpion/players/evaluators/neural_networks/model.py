"""Simple Morpion regressors over handcrafted feature tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from coral.chi_nn import ChiNN
from torch import Tensor, nn

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MORPION_FEATURE_SCHEMA,
    MorpionFeatureSubset,
    full_morpion_feature_subset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_MODEL_KIND,
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
)

MORPION_INPUT_DIM = full_morpion_feature_subset().dimension


@dataclass(frozen=True, slots=True)
class MorpionRegressorArgs:
    """Arguments for Morpion neural evaluator models."""

    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_sizes: tuple[int, ...] | None = None
    graph_max_tokens: int = 1536
    graph_input_feature_dim: int = MORPION_GRAPH_TOKEN_FEATURE_DIM
    graph_d_model: int = 64
    graph_n_head: int = 4
    graph_n_layer: int = 2
    graph_dim_feedforward: int = 256
    graph_dropout_ratio: float = 0.0
    graph_pooling: str = "value_token"
    graph_output_tanh: bool = True

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if self.model_kind == MORPION_GRAPH_MODEL_KIND:
            _validate_graph_transformer_args(self)
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this regressor."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )

    @property
    def input_dim(self) -> int:
        """Return the model input width."""
        if self.model_kind == MORPION_GRAPH_MODEL_KIND:
            return self.graph_input_feature_dim
        return self.feature_subset.dimension


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


class InvalidMorpionGraphRegressorArgsError(ValueError):
    """Raised when graph-transformer regressor args are invalid."""

    @classmethod
    def invalid_input_feature_dim(
        cls,
        expected_dim: int,
    ) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph input feature dimension error."""
        return cls(
            "graph_input_feature_dim must equal "
            f"{expected_dim} for graph_tokens_v1."
        )

    @classmethod
    def invalid_max_tokens(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph max-token-count error."""
        return cls("graph_max_tokens must be >= 2.")

    @classmethod
    def invalid_d_model(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph transformer width error."""
        return cls("graph_d_model must be > 0.")

    @classmethod
    def invalid_n_head(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph attention head count error."""
        return cls("graph_n_head must be > 0.")

    @classmethod
    def incompatible_attention_width(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the incompatible graph attention width error."""
        return cls("graph_d_model must be divisible by graph_n_head.")

    @classmethod
    def invalid_n_layer(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph transformer layer count error."""
        return cls("graph_n_layer must be >= 0.")

    @classmethod
    def invalid_dim_feedforward(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph feedforward width error."""
        return cls("graph_dim_feedforward must be > 0.")

    @classmethod
    def invalid_dropout_ratio(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph dropout ratio error."""
        return cls("graph_dropout_ratio must be >= 0.")

    @classmethod
    def invalid_pooling(cls) -> InvalidMorpionGraphRegressorArgsError:
        """Return the invalid graph pooling mode error."""
        return cls("graph_pooling must be one of {'value_token', 'masked_mean'}.")


def _validate_graph_transformer_args(args: MorpionRegressorArgs) -> None:
    """Validate graph-token model args for the v1 Morpion token schema."""
    if args.graph_input_feature_dim != MORPION_GRAPH_TOKEN_FEATURE_DIM:
        raise InvalidMorpionGraphRegressorArgsError.invalid_input_feature_dim(
            MORPION_GRAPH_TOKEN_FEATURE_DIM
        )
    if args.graph_max_tokens < 2:
        raise InvalidMorpionGraphRegressorArgsError.invalid_max_tokens()
    if args.graph_d_model <= 0:
        raise InvalidMorpionGraphRegressorArgsError.invalid_d_model()
    if args.graph_n_head <= 0:
        raise InvalidMorpionGraphRegressorArgsError.invalid_n_head()
    if args.graph_d_model % args.graph_n_head != 0:
        raise InvalidMorpionGraphRegressorArgsError.incompatible_attention_width()
    if args.graph_n_layer < 0:
        raise InvalidMorpionGraphRegressorArgsError.invalid_n_layer()
    if args.graph_dim_feedforward <= 0:
        raise InvalidMorpionGraphRegressorArgsError.invalid_dim_feedforward()
    if args.graph_dropout_ratio < 0.0:
        raise InvalidMorpionGraphRegressorArgsError.invalid_dropout_ratio()
    if args.graph_pooling not in {"value_token", "masked_mean"}:
        raise InvalidMorpionGraphRegressorArgsError.invalid_pooling()


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
    if args.model_kind == MORPION_GRAPH_MODEL_KIND:
        from coral.neural_networks.models.entity_token_transformer_value_net import (
            EntityTokenTransformerValueNet,
            EntityTokenTransformerValueNetArgs,
        )

        return EntityTokenTransformerValueNet(
            EntityTokenTransformerValueNetArgs(
                input_feature_dim=args.graph_input_feature_dim,
                d_model=args.graph_d_model,
                n_head=args.graph_n_head,
                n_layer=args.graph_n_layer,
                dim_feedforward=args.graph_dim_feedforward,
                dropout_ratio=args.graph_dropout_ratio,
                pooling=args.graph_pooling,  # type: ignore[arg-type]
                output_tanh=args.graph_output_tanh,
            )
        )
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
