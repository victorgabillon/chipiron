"""Defines the neural network model and activation function types."""

from enum import Enum
from typing import Callable

import torch.nn as nn


class NNModelType(str, Enum):
    """Defines the types of neural network models."""

    MULTI_LAYER_PERCEPTRON = "multi_layer_perceptron"
    TRANSFORMER = "transformer"


class ActivationFunctionType(str, Enum):
    """Defines the types of activation functions."""

    TANGENT_HYPERBOLIC = "hyperbolic_tangent"
    PARAMETRIC_RELU = "parametric_relu"
    RELU = "relu"


activation_map: dict[ActivationFunctionType, Callable[[], nn.Module]] = {
    ActivationFunctionType.RELU: nn.ReLU,
    ActivationFunctionType.TANGENT_HYPERBOLIC: nn.Tanh,
    ActivationFunctionType.PARAMETRIC_RELU: nn.PReLU,
}
