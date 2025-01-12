"""
This module provides functionality for managing exploration index managers.

The module includes the following functions and classes:
- create_exploration_index_manager: A function to create an exploration index manager.
- NodeExplorationIndexManager: A class representing a node exploration index manager.
"""

from .factory import create_exploration_index_manager
from .node_exploration_manager import NodeExplorationIndexManager

__all__ = ["create_exploration_index_manager", "NodeExplorationIndexManager"]
