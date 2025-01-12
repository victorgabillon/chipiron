"""
This module provides functionality for managing algorithm node trees.

The module includes classes for creating and managing algorithm node tree managers,
expanding tree structures, and managing tree expansions.

Classes:
- AlgorithmNodeTreeManager: A class for managing algorithm node trees.
- TreeManager: A class for managing tree structures.
- TreeExpansion: A class for representing a single tree expansion.
- TreeExpansions: A class for managing multiple tree expansions.

Functions:
- create_algorithm_node_tree_manager: A function for creating an algorithm node tree manager.

"""

from .algorithm_node_tree_manager import AlgorithmNodeTreeManager
from .factory import create_algorithm_node_tree_manager
from .tree_expander import TreeExpansion, TreeExpansions
from .tree_manager import TreeManager

__all__ = [
    "create_algorithm_node_tree_manager",
    "TreeManager",
    "AlgorithmNodeTreeManager",
    "TreeExpansion",
    "TreeExpansions",
]
