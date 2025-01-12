"""
Module that contains the NeuralNetBoardEvalArgs class.
"""

from dataclasses import dataclass


@dataclass
class NeuralNetBoardEvalArgs:
    """
    Represents the arguments for a neural network board evaluator.

    Attributes:
        nn_type (str): The type of the neural network.
        nn_param_folder_name (str): The name of the folder containing the neural network parameters.
    """

    nn_type: str
    nn_param_folder_name: str
