"""
This module provides a function to create a node factory based on the given node factory name.
"""

from typing import Any

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .base import Base


def create_node_factory(node_factory_name: str) -> Base[Any]:
    """
    Create a node factory based on the given node factory name.

    Args:
        node_factory_name (str): The name of the node factory.

    Returns:
        Base: An instance of the node factory.

    Raises:
        ValueError: If the given node factory name is not implemented.

    """
    tree_node_factory: Base[Any]

    match node_factory_name:
        case "Base_with_tree_node":
            tree_node_factory = Base()
        case "Base_with_algorithm_tree_node":
            tree_node_factory = Base[AlgorithmNode]()
        case other:
            raise ValueError(
                f"please implement your node factory!! no {other} in {__name__}"
            )

    return tree_node_factory
