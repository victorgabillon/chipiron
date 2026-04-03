"""Generic runtime helpers for building NN evaluators from model bundles."""

from __future__ import annotations

from dataclasses import is_dataclass
from enum import Enum
from typing import Any

import dacite

from chipiron.models.model_bundle import ResolvedModelBundle
from chipiron.utils.small_tools import yaml_fetch_args_in_file


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


def load_nn_architecture_args_from_file(architecture_file_path: str) -> Any:
    """Load neural-network architecture args from an explicit YAML file path."""
    architecture_args_cls = _get_neural_net_architecture_args_class()
    args_dict = yaml_fetch_args_in_file(path_file=architecture_file_path)
    cast_types = [Enum]
    if is_dataclass(architecture_args_cls):
        return dacite.from_dict(
            data_class=architecture_args_cls,
            data=args_dict,
            config=dacite.Config(cast=cast_types),
        )
    raise TypeError(
        "Neural network architecture args class must be a dataclass, "
        f"got {architecture_args_cls!r}."
    )


def load_nn_architecture_args_from_bundle(bundle: ResolvedModelBundle) -> Any:
    """Load neural-network architecture args from a resolved model bundle."""
    return load_nn_architecture_args_from_file(bundle.architecture_file_path)


def create_nn_state_eval_from_model_bundle_and_converter(
    bundle: ResolvedModelBundle,
    content_to_input_convert: Any,
) -> Any:
    """Build an NN evaluator from a resolved model bundle and an input converter."""
    nn_architecture_args = load_nn_architecture_args_from_bundle(bundle)
    return _create_nn_state_eval_from_existing_model(
        model_weights_file_name=bundle.weights_file_path,
        nn_architecture_args=nn_architecture_args,
        content_to_input_convert=content_to_input_convert,
    )


__all__ = [
    "create_nn_state_eval_from_model_bundle_and_converter",
    "load_nn_architecture_args_from_bundle",
    "load_nn_architecture_args_from_file",
]
