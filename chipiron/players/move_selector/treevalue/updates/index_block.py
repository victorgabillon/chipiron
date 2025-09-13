"""
This module defines the IndexUpdateInstructionsBlock class, which represents a block of update instructions for
index values in a tree structure.

The IndexUpdateInstructionsBlock class is a dataclass that contains a set of AlgorithmNode objects representing
children with updated index values. It provides methods for merging update instructions and printing information
about the block.
"""

from dataclasses import dataclass, field
from typing import Self

from chipiron.environments.chess_env.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)


@dataclass(slots=True)
class IndexUpdateInstructionsFromOneNode:
    """
    Represents a block of instructions for updating an index.

    Attributes:
        node_sending_update (AlgorithmNode): The node sending the update.
        updated_index (bool): Indicates whether the index has been updated.
    """

    node_sending_update: AlgorithmNode
    updated_index: bool


@dataclass(slots=True)
class IndexUpdateInstructionsTowardsOneParentNode:
    """
    Represents a block of index update instructions intended to a specific node in the algorithm tree.

    This class is used to store and manipulate sets of children with updated index values.

    Attributes:
        moves_with_updated_index (Set[IMove]): A set of children with updated index values.
    """

    moves_with_updated_index: set[moveKey] = field(
        default_factory=lambda: set[moveKey]()
    )

    def add_update_from_one_child_node(
        self,
        update_from_one_child_node: IndexUpdateInstructionsFromOneNode,
        move_from_parent_to_child: moveKey,
    ) -> None:
        """Adds an update from a child node to the parent node.

        Args:
            update_from_one_child_node (IndexUpdateInstructionsFromOneNode): The update instructions from the child node.
            move_from_parent_to_child (moveKey): The move key representing the parent's move to the child.
        """
        if update_from_one_child_node.updated_index:
            self.moves_with_updated_index.add(move_from_parent_to_child)

    def add_update_toward_one_parent_node(self, another_update: Self) -> None:
        """Adds an update from another child node to the parent node.

        Args:
            another_update (Self): The update instructions from another child node.
        """
        self.moves_with_updated_index = (
            self.moves_with_updated_index | another_update.moves_with_updated_index
        )

    def empty(self) -> bool:
        """
        Check if the IndexUpdateInstructionsBlock is empty.

        Returns:
            bool: True if the block is empty, False otherwise.
        """
        empty_bool = not bool(self.moves_with_updated_index)
        return empty_bool

    def print_info(self) -> None:
        """Prints information about the moves with updated indices."""
        print(self.moves_with_updated_index)
