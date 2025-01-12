"""
This module defines the different types of node evaluators used in the tree value calculation.
"""

from enum import Enum


class NodeEvaluatorTypes(str, Enum):
    """
    Enum class representing different types of node evaluators.
    """

    NeuralNetwork = "neural_network"
    BasicEvaluation = "basic_evaluation"
