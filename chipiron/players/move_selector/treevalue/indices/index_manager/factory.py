"""
This module contains the factory function to create a node exploration index manager based on the given index computation
"""

from chipiron.players.move_selector.treevalue.indices.index_manager.node_exploration_manager import (
    NodeExplorationIndexManager,
    NullNodeExplorationIndexManager,
    UpdateIndexGlobalMinChange,
    UpdateIndexLocalMinChange,
    UpdateIndexZipfFactoredProba,
)
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import (
    IndexComputationType,
)


def create_exploration_index_manager(
    index_computation: IndexComputationType | None = None,
) -> NodeExplorationIndexManager:
    """
    Creates a node exploration index manager based on the given index computation type.

    Args:
        index_computation (IndexComputationType | None): The type of index computation to be used.
        Defaults to None.

    Returns:
        NodeExplorationIndexManager: The created node exploration index manager.

    Raises:
        ValueError: If the given index computation type is not found.

    """
    node_exploration_manager: NodeExplorationIndexManager
    if index_computation is None:
        node_exploration_manager = NullNodeExplorationIndexManager()
    else:
        match index_computation:
            case IndexComputationType.MinGlobalChange:
                node_exploration_manager = UpdateIndexGlobalMinChange()
            case IndexComputationType.RecurZipf:
                node_exploration_manager = UpdateIndexZipfFactoredProba()
            case IndexComputationType.MinLocalChange:
                node_exploration_manager = UpdateIndexLocalMinChange()
            case other:
                raise ValueError(f"player creator: can not find {other} in {__name__}")

    return node_exploration_manager
