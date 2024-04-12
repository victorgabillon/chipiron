"""
This module defines the IndexUpdateInstructionsBlock class, which represents a block of update instructions for
index values in a tree structure.

The IndexUpdateInstructionsBlock class is a dataclass that contains a set of AlgorithmNode objects representing
children with updated index values. It provides methods for merging update instructions and printing information
about the block.
"""

from dataclasses import dataclass, field
from typing import Set

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode


@dataclass(slots=True)
class IndexUpdateInstructionsBlock:
    """
    Represents a block of index update instructions.

    This class is used to store and manipulate sets of children with updated index values.

    Attributes:
        children_with_updated_index (Set[AlgorithmNode]): A set of children with updated index values.
    """

    children_with_updated_index: Set[AlgorithmNode] = field(default_factory=set)

    def merge(
            self,
            an_update_instruction: 'IndexUpdateInstructionsBlock',
            another_update_instruction: 'IndexUpdateInstructionsBlock'
    ) -> None:
        """
        Merge two update instructions blocks by combining their sets of children with updated index values.

        Args:
            an_update_instruction (IndexUpdateInstructionsBlock): The first update instruction block to merge.
            another_update_instruction (IndexUpdateInstructionsBlock): The second update instruction block to merge.

        Returns:
            None
        """
        self.children_with_updated_index = an_update_instruction.children_with_updated_index | another_update_instruction.children_with_updated_index

    def print_info(self) -> None:
        """
        Print information about the IndexUpdateInstructionsBlock.

        This method prints the number of children with updated index values and their IDs.

        Returns:
            None
        """
        print('upInstructions printing')
        print(len(self.children_with_updated_index), 'children_with_updated_index', end=' ')
        for child in self.children_with_updated_index:
            print(child.id, end=' ')
        print()

    def empty(self) -> bool:
        """
        Check if the IndexUpdateInstructionsBlock is empty.

        Returns:
            bool: True if the block is empty, False otherwise.
        """
        empty_bool = not bool(self.children_with_updated_index)
        return empty_bool
