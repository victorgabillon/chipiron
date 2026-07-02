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
        graph_max_tokens=args.graph_max_tokens,
        graph_input_feature_dim=args.graph_input_feature_dim,
        graph_d_model=args.graph_d_model,
        graph_n_head=args.graph_n_head,
        graph_n_layer=args.graph_n_layer,
        graph_dim_feedforward=args.graph_dim_feedforward,
        graph_dropout_ratio=args.graph_dropout_ratio,
        graph_pooling=args.graph_pooling,
        graph_output_tanh=args.graph_output_tanh,
    )
