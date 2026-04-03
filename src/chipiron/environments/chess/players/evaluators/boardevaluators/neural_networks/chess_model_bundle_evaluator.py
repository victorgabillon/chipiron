"""Chess-specific bundle helpers for NN evaluator construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks.chipiron_nn_args import (
    create_chess_content_to_input_from_bundle,
)
from chipiron.models.model_bundle import ResolvedModelBundle
from chipiron.players.boardevaluators.neural_networks.model_bundle_runtime import (
    create_nn_state_eval_from_model_bundle_and_converter,
)

if TYPE_CHECKING:
    from coral.neural_networks import NNBWStateEvaluator

    from chipiron.environments.chess.types import ChessState


def create_chess_nn_state_eval_from_model_bundle(
    bundle: ResolvedModelBundle,
) -> NNBWStateEvaluator[ChessState]:
    """Build a chess NN evaluator from a resolved model bundle."""
    content_to_input_convert = create_chess_content_to_input_from_bundle(bundle)
    return cast(
        "NNBWStateEvaluator[ChessState]",
        create_nn_state_eval_from_model_bundle_and_converter(
            bundle,
            content_to_input_convert,
        ),
    )


__all__ = ["create_chess_nn_state_eval_from_model_bundle"]
