"""
This module defines the ValueUpdateInstructionsBlock class and a helper function to create instances of it.

The ValueUpdateInstructionsBlock class represents a block of update instructions for a tree value node in
 a move selector algorithm. It contains sets of children nodes that have been updated with new values,
  best moves, or are newly over.

The create_value_update_instructions_block function is a helper function that creates an instance of
 the ValueUpdateInstructionsBlock class with the specified update instructions.

"""

from dataclasses import dataclass, field
from typing import Set

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode


@dataclass(slots=True)
class ValueUpdateInstructionsBlock:
    """
    Represents a block of update instructions for values in the algorithm tree.

    Attributes:
        children_with_updated_over (Set[AlgorithmNode]): Set of children nodes with updated 'over' value.
        children_with_updated_value (Set[AlgorithmNode]): Set of children nodes with updated 'value' value.
        children_with_updated_best_move (Set[AlgorithmNode]): Set of children nodes with updated 'best_move' value.
    """

    children_with_updated_over: Set[AlgorithmNode] = field(default_factory=set)
    children_with_updated_value: Set[AlgorithmNode] = field(default_factory=set)
    children_with_updated_best_move: Set[AlgorithmNode] = field(default_factory=set)

    def merge(
            self,
            an_update_instruction: 'ValueUpdateInstructionsBlock',
            another_update_instruction: 'ValueUpdateInstructionsBlock'
    ) -> None:
        """
        Merge two update instructions blocks into the current block.

        Args:
            an_update_instruction (ValueUpdateInstructionsBlock): The first update instructions block to merge.
            another_update_instruction (ValueUpdateInstructionsBlock): The second update instructions block to merge.

        Returns:
            None
        """
        self.children_with_updated_value = (an_update_instruction.children_with_updated_value
                                            | another_update_instruction.children_with_updated_value)
        self.children_with_updated_best_move = (an_update_instruction.children_with_updated_best_move
                                                | another_update_instruction.children_with_updated_best_move)
        self.children_with_updated_over = (an_update_instruction.children_with_updated_over
                                           | another_update_instruction.children_with_updated_over)

    def print_info(self) -> None:
        """
        Print information about the update instructions block.

        Returns:
            None
        """
        print('upInstructions printing')
        print(len(self.children_with_updated_value), 'children_with_updated_value', end=' ')
        for child in self.children_with_updated_value:
            print(child.id, end=' ')
        print('\n', len(self.children_with_updated_best_move), 'children_with_updated_best_move:', end=' ')
        for child in self.children_with_updated_best_move:
            print(child.id, end=' ')
        print('\n', len(self.children_with_updated_over), 'children_with_updated_over', end=' ')
        for child in self.children_with_updated_over:
            print(child.id, end=' ')
        print()

    def empty(self) -> bool:
        """
        Check if all the components of the update instructions block are empty.

        Returns:
            bool: True if all components are empty, False otherwise.
        """
        empty_bool = not bool(self.children_with_updated_value) and not bool(
            self.children_with_updated_best_move) and not bool(self.children_with_updated_over)
        return empty_bool


def create_value_update_instructions_block(
        node_sending_update: AlgorithmNode,  # node(or None)
        is_node_newly_over: bool,  # boolean
        new_value_for_node: bool,  # boolean
        new_best_move_for_node: bool  # boolean
) -> ValueUpdateInstructionsBlock:
    """
    Create an instance of ValueUpdateInstructionsBlock with the specified update instructions.

    Args:
        node_sending_update: The node sending the update (or None).
        is_node_newly_over: A boolean indicating if the node is newly over.
        new_value_for_node: A boolean indicating if there is a new value for the node.
        new_best_move_for_node: A boolean indicating if there is a new best move for the node.

    Returns:
        ValueUpdateInstructionsBlock: An instance of ValueUpdateInstructionsBlock with the specified update instructions.
    """
    value_update_instructions_block = ValueUpdateInstructionsBlock(
        children_with_updated_over={node_sending_update} if is_node_newly_over else set(),
        children_with_updated_value={node_sending_update} if new_value_for_node else set(),
        children_with_updated_best_move={node_sending_update} if new_best_move_for_node else set()
    )
    return value_update_instructions_block
