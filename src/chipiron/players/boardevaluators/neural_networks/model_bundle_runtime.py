"""Generic runtime helpers for building NN evaluators from model bundles."""


from enum import Enum
from typing import TYPE_CHECKING, Any

import dacite  # pyright: ignore[reportMissingImports]

from chipiron.models.model_bundle import ResolvedModelBundle
from chipiron.utils.small_tools import yaml_fetch_args_in_file

if TYPE_CHECKING:
    from coral.neural_networks.neural_net_architecture_args import (
        NeuralNetArchitectureArgs,
    )


def load_nn_architecture_args_from_file(
    architecture_file_path: str,
) -> "NeuralNetArchitectureArgs":
    """Load neural-network architecture args from an explicit YAML file path."""
    from coral.neural_networks.neural_net_architecture_args import (
        NeuralNetArchitectureArgs,
    )  # pyright: ignore[reportMissingImports]

    args_dict = yaml_fetch_args_in_file(path_file=architecture_file_path)
    return dacite.from_dict(
        data_class=NeuralNetArchitectureArgs,
        data=args_dict,
        config=dacite.Config(cast=[Enum]),
    )


def load_nn_architecture_args_from_bundle(
    bundle: ResolvedModelBundle,
) -> "NeuralNetArchitectureArgs":
    """Load neural-network architecture args from a resolved model bundle."""
    return load_nn_architecture_args_from_file(bundle.architecture_file_path)


def create_nn_state_eval_from_model_bundle_and_converter(
    bundle: ResolvedModelBundle,
    content_to_input_convert: Any,
) -> Any:
    """Build an NN evaluator from a resolved model bundle and an input converter."""
    from coral.neural_networks.factory import (
        create_nn_state_eval_from_nn_parameters_file_and_existing_model,
    )  # pyright: ignore[reportMissingImports]

    nn_architecture_args = load_nn_architecture_args_from_bundle(bundle)
    return create_nn_state_eval_from_nn_parameters_file_and_existing_model(
        model_weights_file_name=bundle.weights_file_path,
        nn_architecture_args=nn_architecture_args,
        content_to_input_convert=content_to_input_convert,
    )


__all__ = [
    "create_nn_state_eval_from_model_bundle_and_converter",
    "load_nn_architecture_args_from_bundle",
    "load_nn_architecture_args_from_file",
]
