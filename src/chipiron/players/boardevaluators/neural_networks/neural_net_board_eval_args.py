"""Bundle-first configuration for neural-network board evaluators."""


from dataclasses import dataclass
from typing import Literal

from chipiron.models.model_bundle import ModelBundleRef

NN_NET_EVAL_LITERAL_STRING: Literal["neural_network"] = "neural_network"
NN_NET_EVAL_STRING: str = NN_NET_EVAL_LITERAL_STRING


@dataclass(frozen=True, slots=True)
class NeuralNetModelBundleArgs:
    """Public NN artifact configuration for evaluator YAML files."""

    model_bundle: ModelBundleRef


@dataclass(frozen=True, slots=True)
class NeuralNetBoardEvalArgs:
    """Arguments for a neural-network board evaluator backed by a model bundle."""

    neural_nets_model_and_architecture: NeuralNetModelBundleArgs
    type: Literal["neural_network"] = NN_NET_EVAL_LITERAL_STRING


__all__ = [
    "NN_NET_EVAL_LITERAL_STRING",
    "NN_NET_EVAL_STRING",
    "NeuralNetBoardEvalArgs",
    "NeuralNetModelBundleArgs",
]
