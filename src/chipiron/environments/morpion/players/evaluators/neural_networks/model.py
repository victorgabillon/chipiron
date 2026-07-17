"""Simple Morpion regressors over handcrafted feature tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from coral.chi_nn import ChiNN
from torch import Tensor, nn

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
    is_morpion_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MORPION_FEATURE_SCHEMA,
    MorpionFeatureSubset,
    full_morpion_feature_subset,
    resolve_morpion_feature_subset,
)

MORPION_INPUT_DIM = full_morpion_feature_subset().dimension


@dataclass(frozen=True, slots=True)
class MorpionRegressorArgs:
    """Arguments for Morpion neural evaluator models."""

    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_sizes: tuple[int, ...] | None = None
    entity_max_tokens: int = 1536
    entity_input_feature_dim: int = MORPION_ENTITY_TOKEN_FEATURE_DIM
    entity_d_model: int = 64
    entity_n_head: int = 4
    entity_n_layer: int = 2
    entity_dim_feedforward: int = 256
    entity_dropout_ratio: float = 0.0
    entity_pooling: str = "value_token"
    entity_output_tanh: bool = False
    entity_use_validity_feature: bool = True
    entity_relation_schema: str | None = None
    entity_relation_type_count: int | None = None

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if is_morpion_entity_token_model_kind(
            self.model_kind
        ) or is_relational_entity_token_model_kind(self.model_kind):
            _validate_entity_token_transformer_value_net_args(self)
        if is_relational_entity_token_model_kind(self.model_kind):
            _validate_relational_entity_token_transformer_value_net_args(self)
        elif (
            self.entity_relation_schema is not None
            or self.entity_relation_type_count is not None
        ):
            raise InvalidMorpionEntityTokenRegressorArgsError.unexpected_relation_metadata()
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
        if is_morpion_entity_token_model_kind(
            self.model_kind
        ) or is_relational_entity_token_model_kind(self.model_kind):
            return self.entity_input_feature_dim
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


class InvalidMorpionEntityTokenRegressorArgsError(ValueError):
    """Raised when entity-token regressor args are invalid."""

    @classmethod
    def invalid_input_feature_dim(
        cls,
        expected_dim: int,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity input feature dimension error."""
        return cls(
            "entity_input_feature_dim must equal "
            f"{expected_dim} for {MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION}."
        )

    @classmethod
    def invalid_max_tokens(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity max-token-count error."""
        return cls("entity_max_tokens must be >= 1.")

    @classmethod
    def invalid_d_model(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity transformer width error."""
        return cls("entity_d_model must be > 0.")

    @classmethod
    def invalid_n_head(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity attention head count error."""
        return cls("entity_n_head must be > 0.")

    @classmethod
    def incompatible_attention_width(
        cls,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the incompatible entity attention width error."""
        return cls("entity_d_model must be divisible by entity_n_head.")

    @classmethod
    def invalid_n_layer(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity transformer layer count error."""
        return cls("entity_n_layer must be >= 0.")

    @classmethod
    def invalid_dim_feedforward(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity feedforward width error."""
        return cls("entity_dim_feedforward must be > 0.")

    @classmethod
    def invalid_dropout_ratio(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity dropout ratio error."""
        return cls("entity_dropout_ratio must be >= 0.")

    @classmethod
    def invalid_pooling(cls) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid entity pooling mode error."""
        return cls("entity_pooling must be one of {'value_token', 'masked_mean'}.")

    @classmethod
    def invalid_validity_feature(
        cls,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid validity-feature configuration error."""
        return cls("entity_use_validity_feature must be true.")

    @classmethod
    def invalid_relation_schema(
        cls,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid relation-schema configuration error."""
        return cls(
            "entity_relation_schema must equal "
            f"{MORPION_ENTITY_RELATION_SCHEMA!r} for the relational model."
        )

    @classmethod
    def invalid_relation_type_count(
        cls,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the invalid relation-count configuration error."""
        return cls(
            "entity_relation_type_count must equal "
            f"{MORPION_ENTITY_RELATION_TYPE_COUNT} for the relational model."
        )

    @classmethod
    def unexpected_relation_metadata(
        cls,
    ) -> InvalidMorpionEntityTokenRegressorArgsError:
        """Return the relation-metadata-on-non-relational-model error."""
        return cls("Relation metadata is only valid for the relational model kind.")


def _validate_entity_token_transformer_value_net_args(
    args: MorpionRegressorArgs,
) -> None:
    """Validate entity-token model args for the v1 Morpion token schema."""
    if args.entity_input_feature_dim != MORPION_ENTITY_TOKEN_FEATURE_DIM:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_input_feature_dim(
            MORPION_ENTITY_TOKEN_FEATURE_DIM
        )
    if args.entity_max_tokens < 1:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_max_tokens()
    if args.entity_d_model <= 0:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_d_model()
    if args.entity_n_head <= 0:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_n_head()
    if args.entity_d_model % args.entity_n_head != 0:
        raise InvalidMorpionEntityTokenRegressorArgsError.incompatible_attention_width()
    if args.entity_n_layer < 0:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_n_layer()
    if args.entity_dim_feedforward <= 0:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_dim_feedforward()
    if args.entity_dropout_ratio < 0.0:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_dropout_ratio()
    if args.entity_pooling not in {"value_token", "masked_mean"}:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_pooling()
    if not args.entity_use_validity_feature:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_validity_feature()


def _validate_relational_entity_token_transformer_value_net_args(
    args: MorpionRegressorArgs,
) -> None:
    """Validate Morpion relation metadata against the v1 relation schema."""
    if args.entity_relation_schema != MORPION_ENTITY_RELATION_SCHEMA:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_relation_schema()
    if args.entity_relation_type_count != MORPION_ENTITY_RELATION_TYPE_COUNT:
        raise InvalidMorpionEntityTokenRegressorArgsError.invalid_relation_type_count()


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
            layers.extend([
                nn.Linear(previous_dim, hidden_size),
                nn.ReLU(),
            ])
            previous_dim = hidden_size
        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers)
    if is_morpion_entity_token_model_kind(
        args.model_kind
    ) or is_relational_entity_token_model_kind(args.model_kind):
        from coral.neural_networks.factory import (
            create_nn,  # pylint: disable=import-outside-toplevel
        )
        from coral.neural_networks.models.entity_token_transformer_value_net import (  # pylint: disable=import-outside-toplevel
            EntityTokenTransformerValueNetArgs,
        )
        from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (  # pylint: disable=import-outside-toplevel
            RelationBiasedEntityTokenTransformerValueNetArgs,
        )

        common_args = {
            "input_feature_dim": args.entity_input_feature_dim,
            "d_model": args.entity_d_model,
            "n_head": args.entity_n_head,
            "n_layer": args.entity_n_layer,
            "dim_feedforward": args.entity_dim_feedforward,
            "dropout_ratio": args.entity_dropout_ratio,
            "pooling": args.entity_pooling,
            "output_tanh": args.entity_output_tanh,
            "use_value_token": True,
            "use_validity_feature": args.entity_use_validity_feature,
            "validity_feature_index": -1,
        }
        if is_relational_entity_token_model_kind(args.model_kind):
            model_type_args = RelationBiasedEntityTokenTransformerValueNetArgs(
                **common_args,  # type: ignore[arg-type]
                num_relation_types=cast("int", args.entity_relation_type_count),
            )
        else:
            model_type_args = EntityTokenTransformerValueNetArgs(
                **common_args,  # type: ignore[arg-type]
            )
        return create_nn(model_type_args)
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

    def forward(
        self,
        input_tensor: Tensor,
        *auxiliary_input_tensors: Tensor,
    ) -> Tensor:
        """Forward every positional model input to the Coral model."""
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        return cast(
            "Tensor",
            self.net(input_tensor, *auxiliary_input_tensors),
        )

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
