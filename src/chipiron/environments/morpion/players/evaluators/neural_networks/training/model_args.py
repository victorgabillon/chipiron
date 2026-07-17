"""Build Morpion regressor construction args from training args."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
)

if TYPE_CHECKING:
    from .args import MorpionTrainingArgs


def resolve_hidden_sizes(args: MorpionTrainingArgs) -> tuple[int, ...] | None:
    """Resolve legacy and current hidden-layer arguments into one tuple."""
    if args.hidden_sizes is not None:
        return args.hidden_sizes
    if args.hidden_dim is not None:
        return (args.hidden_dim,)
    return None


def morpion_regressor_args_from_training_args(
    args: MorpionTrainingArgs,
) -> MorpionRegressorArgs:
    """Build model construction args from Morpion training args."""
    return MorpionRegressorArgs(
        model_kind=args.model_kind,
        feature_subset_name=args.feature_subset_name,
        feature_names=args.feature_names,
        hidden_sizes=resolve_hidden_sizes(args),
        entity_max_tokens=args.entity_max_tokens,
        entity_input_feature_dim=args.entity_input_feature_dim,
        entity_d_model=args.entity_d_model,
        entity_n_head=args.entity_n_head,
        entity_n_layer=args.entity_n_layer,
        entity_dim_feedforward=args.entity_dim_feedforward,
        entity_dropout_ratio=args.entity_dropout_ratio,
        entity_pooling=args.entity_pooling,
        entity_output_tanh=args.entity_output_tanh,
        entity_use_validity_feature=args.entity_use_validity_feature,
        entity_relation_schema=args.entity_relation_schema,
        entity_relation_type_count=args.entity_relation_type_count,
    )
