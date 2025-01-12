"""
This module provides functionality for creating tree and value builders for the TreeAndValuePlayer.

The TreeAndValuePlayerArgs class represents the arguments for configuring the TreeAndValuePlayer.
The create_tree_and_value_builders function is used to create the tree and value builders.

Example usage:
    from chipiron.players.move_selector.treevalue import TreeAndValuePlayerArgs, create_tree_and_value_builders

    args = TreeAndValuePlayerArgs(...)
    builders = create_tree_and_value_builders(args)

    # Use the builders to create the TreeAndValuePlayer
    player = TreeAndValuePlayer(builders.tree_builder, builders.value_builder)
"""

from .factory import TreeAndValuePlayerArgs, create_tree_and_value_builders

__all__ = ["TreeAndValuePlayerArgs", "create_tree_and_value_builders"]
