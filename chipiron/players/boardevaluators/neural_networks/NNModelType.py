from enum import Enum
from typing import Callable

import torch.nn as nn


class NNModelType(str, Enum):
    MultiLayerPerceptron = "multi_layer_perceptron"
    Transformer = "transformer"


class ActivationFunctionType(str, Enum):
    TangentHyperbolic = "hyperbolic_tangent"
    ParametricRelu = "parametric_relu"
    Relu = "relu"


activation_map: dict[ActivationFunctionType, Callable[[], nn.Module]] = {
    ActivationFunctionType.Relu: nn.ReLU,
    ActivationFunctionType.TangentHyperbolic: nn.Tanh,
    ActivationFunctionType.ParametricRelu: nn.PReLU,
}
