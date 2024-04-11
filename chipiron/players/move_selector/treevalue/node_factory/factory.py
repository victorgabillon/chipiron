"""
This module provides a function to create a node factory based on the given node factory name.
"""

from .base import Base


def create_node_factory(
        node_factory_name: str
) -> Base:
    """
    Create a node factory based on the given node factory name.

    Args:
        node_factory_name (str): The name of the node factory.

    Returns:
        Base: An instance of the node factory.

    Raises:
        ValueError: If the given node factory name is not implemented.

    """
    tree_node_factory: Base

    match node_factory_name:
        case 'Base':
            tree_node_factory = Base()
        case other:
            raise ValueError(f'please implement your node factory!! no {other} in {__name__}')

    return tree_node_factory
