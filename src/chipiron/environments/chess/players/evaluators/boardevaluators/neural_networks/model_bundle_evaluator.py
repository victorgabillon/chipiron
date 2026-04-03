"""Bundle-based construction helpers for chess neural-network evaluators."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import dacite

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks.chipiron_nn_args import (
    create_content_to_input_from_bundle,
)
from chipiron.models.model_bundle import ResolvedModelBundle
from chipiron.utils.small_tools import yaml_fetch_args_in_file

if TYPE_CHECKING:
    from coral.neural_networks import NNBWStateEvaluator
    from coral.neural_networks.neural_net_architecture_args import (
        NeuralNetArchitectureArgs,
    )

    from chipiron.environments.chess.types import ChessState


def _get_neural_net_architecture_args_class() -> type[Any]:
    """Return the coral architecture-args class lazily for easier testing."""
    from coral.neural_networks.neural_net_architecture_args import (
        NeuralNetArchitectureArgs,
    )

    return NeuralNetArchitectureArgs


def _create_nn_state_eval_from_existing_model(
    *,
    model_weights_file_name: str,
    nn_architecture_args: Any,
    content_to_input_convert: Any,
) -> Any:
    """Call coral's NN evaluator factory lazily for easier testing."""
    from coral.neural_networks.factory import (
        create_nn_state_eval_from_nn_parameters_file_and_existing_model,
    )

    return create_nn_state_eval_from_nn_parameters_file_and_existing_model(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
        content_to_input_convert=content_to_input_convert,
    )


def load_nn_architecture_args_from_file(
    architecture_file_path: str,
) -> NeuralNetArchitectureArgs:
    """Load neural-network architecture args from an explicit YAML file path."""
    architecture_args_cls = _get_neural_net_architecture_args_class()
    args_dict = yaml_fetch_args_in_file(path_file=architecture_file_path)
    return cast(
        "NeuralNetArchitectureArgs",
        dacite.from_dict(
            data_class=architecture_args_cls,
            data=args_dict,
            config=dacite.Config(cast=[Enum]),
        ),
    )


def load_nn_architecture_args_from_bundle(
    bundle: ResolvedModelBundle,
) -> NeuralNetArchitectureArgs:
    """Load neural-network architecture args from a resolved model bundle."""
    return load_nn_architecture_args_from_file(bundle.architecture_file_path)


def create_nn_state_eval_from_model_bundle(
    bundle: ResolvedModelBundle,
) -> NNBWStateEvaluator[ChessState]:
    """Build a chess NN evaluator from a resolved model bundle."""
    nn_architecture_args = load_nn_architecture_args_from_bundle(bundle)
    content_to_input_convert = create_content_to_input_from_bundle(bundle)
    return cast(
        "NNBWStateEvaluator[ChessState]",
        _create_nn_state_eval_from_existing_model(
            model_weights_file_name=bundle.weights_file_path,
            nn_architecture_args=nn_architecture_args,
            content_to_input_convert=content_to_input_convert,
        ),
    )


__all__ = [
    "create_nn_state_eval_from_model_bundle",
    "load_nn_architecture_args_from_bundle",
    "load_nn_architecture_args_from_file",
]
