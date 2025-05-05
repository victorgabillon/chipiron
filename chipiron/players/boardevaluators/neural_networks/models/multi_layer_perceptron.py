"""
This module defines the neural network model NetPP2D2_2_PRELU.

The NetPP2D2_2_PRELU class is a subclass of ChiNN and implements the forward pass of the neural network.
It consists of two fully connected layers with PReLU activation functions.


Methods:
    __init__(): Initializes the NetPP2D2_2_PRELU model.
    forward(x: torch.Tensor) -> torch.Tensor: Performs the forward pass of the neural network.
    init_weights(file: str) -> None: Initializes the weights of the model.
    print_param() -> None: Prints the parameters of the model.
    print_input(input: torch.Tensor) -> None: Prints the input tensor.

Helper Functions:
    print_piece_param(i: int, vec: torch.Tensor) -> None: Prints the parameters of a specific piece.

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal

import torch
import torch.nn as nn
import yaml

from chipiron.players.boardevaluators.neural_networks.NNModelType import (
    ActivationFunctionType,
    NNModelType,
    activation_map,
)
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.logger import chipiron_logger


@dataclass
class MultiLayerPerceptronArgs:
    type: Literal[NNModelType.MultiLayerPerceptron]
    number_neurons_per_layer: list[int]
    list_of_activation_functions: list[ActivationFunctionType]

    def __str__(self) -> str:
        neurons = "-".join(map(str, self.number_neurons_per_layer))
        activations = "-".join(act.value for act in self.list_of_activation_functions)
        return f"Type: {self.type}, Neurons: [{neurons}], Activations: [{activations}]"

    def filename(self) -> str:
        # Create a filename-safe version by replacing spaces with underscores and ensuring everything is lowercase
        neurons = "_".join(map(str, self.number_neurons_per_layer))
        activations = "_".join(act.value for act in self.list_of_activation_functions)
        return f"{self.type}_{neurons}_{activations}"


def build_sequential(
    layers: list[int], activations: list[ActivationFunctionType]
) -> nn.Sequential:
    modules: list[nn.Module] = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        act: ActivationFunctionType = activations[i]
        if act in activation_map:
            modules.append(activation_map[act]())
    return nn.Sequential(*modules)


def extract_sequential_model_data(model: nn.Sequential) -> Dict[str, Any]:
    """
    Extracts the weights and biases from a Sequential model in a layered dictionary format.

    Args:
        model (nn.Sequential): The Sequential model to extract from.

    Returns:
        Dict[str, Any]: A dictionary with layer names as keys and parameter data as values.
    """
    layers_data = {}

    for idx, layer in enumerate(model):
        layer_info = {}
        if hasattr(layer, "weight") and layer.weight is not None:
            layer_info["weight"] = layer.weight.detach().cpu().numpy().tolist()
        if hasattr(layer, "bias") and layer.bias is not None:
            layer_info["bias"] = layer.bias.detach().cpu().numpy().tolist()

        layer_name = f"layer_{idx}_{layer.__class__.__name__}"
        layers_data[layer_name] = layer_info if layer_info else "No parameters"

    return layers_data


class MultiLayerPerceptron(ChiNN):
    """
    Neural network model for board evaluation using 2D2_2_PRELU architecture.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        relu_1 (nn.PReLU): The first PReLU activation function.
        fc2 (nn.Linear): The second fully connected layer.
        tanh (nn.Tanh): The tanh activation function.
    """

    def __init__(self, args: MultiLayerPerceptronArgs) -> None:
        """Constructor for the NetPP2D2_2_PRELU class. Initializes the neural network layers."""
        super(MultiLayerPerceptron, self).__init__()

        self.model: Callable[[torch.Tensor], torch.Tensor] = build_sequential(
            layers=args.number_neurons_per_layer,
            activations=args.list_of_activation_functions,
        )

    def __getstate__(self) -> None:
        """Get the state of the neural network."""
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        y = self.model(x)
        return y

    def init_weights(self) -> None:
        """Initialize the weights of the neural network."""
        pass

    def print_input(self, input: torch.Tensor) -> None:
        """
        Print the input tensor.

        Args:
            input (torch.Tensor): The input tensor.
        """

    ...

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """
        Writes the model weights and biases into a YAML file.

        Args:
            file_path (str): The path where the YAML file will be saved.
        """
        assert isinstance(self.model, nn.Sequential)
        layers_data = extract_sequential_model_data(self.model)

        with open(file_path, "w") as f:
            yaml.dump(layers_data, f, default_flow_style=False, sort_keys=False)

        chipiron_logger.info(f"Model weights successfully written to {file_path}")

    def print_param(self) -> None:
        """
        Prints the model weights and biases to the console in YAML format.

        Args:
            model (nn.Sequential): The Sequential model.
        """
        assert isinstance(self.model, nn.Sequential)
        layers_data = extract_sequential_model_data(self.model)
        yaml_str = yaml.dump(layers_data, default_flow_style=False, sort_keys=False)
        chipiron_logger.info(yaml_str)


def print_piece_param(i: int, vec: torch.Tensor) -> None:
    """
    Prints the piece parameter at index i from the given tensor.

    Args:
        i (int): The index of the piece parameter to print.
        vec (torch.Tensor): The tensor containing the piece parameters.

    Returns:
        None
    """
    ...
