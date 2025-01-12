"""
This module provides functionality for evaluating nodes in a tree structure.

The module includes a factory function for creating node evaluators, as well as classes for representing
node evaluators, evaluation queries, and node evaluator arguments.

Available objects:
- AllNodeEvaluatorArgs: A named tuple representing all the arguments for creating a node evaluator.
- NodeEvaluator: A class representing a node evaluator.
- create_node_evaluator: A factory function for creating a node evaluator.
- EvaluationQueries: An enumeration representing different types of evaluation queries.
- NodeEvaluatorArgs: A class representing the arguments for creating a node evaluator.
"""

from .factory import AllNodeEvaluatorArgs, create_node_evaluator
from .node_evaluator import EvaluationQueries, NodeEvaluator
from .node_evaluator_args import NodeEvaluatorArgs

__all__ = [
    "AllNodeEvaluatorArgs",
    "NodeEvaluator",
    "create_node_evaluator",
    "EvaluationQueries",
    "NodeEvaluatorArgs",
]
