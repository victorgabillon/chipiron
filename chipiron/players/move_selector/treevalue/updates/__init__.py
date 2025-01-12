"""
This module provides classes and functions for updating tree values in the move selector.

Classes:
- AlgorithmNodeUpdater: A class for updating algorithm nodes in the tree.
- MinMaxEvaluationUpdater: A class for updating min-max evaluation values in the tree.

Functions:
- create_algorithm_node_updater: A function for creating an instance of AlgorithmNodeUpdater.

Other:
- UpdateInstructions: A class representing update instructions for a single node.
- UpdateInstructionsBatch: A class representing a batch of update instructions.

"""

from .algorithm_node_updater import AlgorithmNodeUpdater
from .factory import create_algorithm_node_updater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater
from .updates_file import (
    UpdateInstructionsFromOneNode,
    UpdateInstructionsTowardsMultipleNodes,
    UpdateInstructionsTowardsOneParentNode,
)

__all__ = [
    "create_algorithm_node_updater",
    "AlgorithmNodeUpdater",
    "UpdateInstructionsFromOneNode",
    "UpdateInstructionsTowardsOneParentNode",
    "MinMaxEvaluationUpdater",
    "UpdateInstructionsTowardsMultipleNodes",
]
