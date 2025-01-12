"""
This module defines the ValueUpdateInstructionsBlock class and a helper function to create instances of it.

The ValueUpdateInstructionsBlock class represents a block of update instructions for a tree value node in
 a move selector algorithm. It contains sets of moves that have been updated with new values,
  best moves, or are newly over.

The create_value_update_instructions_block function is a helper function that creates an instance of
 the ValueUpdateInstructionsBlock class with the specified update instructions.

"""

from dataclasses import dataclass, field
from typing import Self

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)


@dataclass(slots=True)
class ValueUpdateInstructionsFromOneNode:
    node_sending_update: AlgorithmNode
    is_node_newly_over: bool
    new_value_for_node: bool
    new_best_move_for_node: bool


@dataclass(slots=True)
class ValueUpdateInstructionsTowardsOneParentNode:
    """
    Represents a block of value-update instructions intended to a specific node in the algorithm tree.

    Attributes:
        moves_with_updated_over (Set[AlgorithmNode]): Set of moves with updated 'over' value.
        moves_with_updated_value (Set[AlgorithmNode]): Set of moves with updated 'value' value.
        moves_with_updated_best_move (Set[AlgorithmNode]): Set of moves with updated 'best_move' value.
    """

    moves_with_updated_over: set[moveKey] = field(default_factory=set)
    moves_with_updated_value: set[moveKey] = field(default_factory=set)
    moves_with_updated_best_move: set[moveKey] = field(default_factory=set)

    def add_update_from_one_child_node(
        self,
        update_from_one_child_node: ValueUpdateInstructionsFromOneNode,
        move_from_parent_to_child: moveKey,
    ) -> None:
        if update_from_one_child_node.is_node_newly_over:
            self.moves_with_updated_over.add(move_from_parent_to_child)
        if update_from_one_child_node.new_value_for_node:
            self.moves_with_updated_value.add(move_from_parent_to_child)
        if update_from_one_child_node.new_best_move_for_node:
            self.moves_with_updated_best_move.add(move_from_parent_to_child)

    def add_update_toward_one_parent_node(self, another_update: Self) -> None:
        self.moves_with_updated_value = (
            self.moves_with_updated_value | another_update.moves_with_updated_over
        )
        self.moves_with_updated_over = (
            self.moves_with_updated_over | another_update.moves_with_updated_over
        )
        self.moves_with_updated_best_move = (
            self.moves_with_updated_best_move
            | another_update.moves_with_updated_best_move
        )

    def print_info(self) -> None:
        """
        Print information about the update instructions block.

        Returns:
            None
        """
        print("upInstructions printing")
        print(len(self.moves_with_updated_value), "moves_with_updated_value", end=" ")
        for move in self.moves_with_updated_value:
            print(move, end=" ")
        print(
            "\n",
            len(self.moves_with_updated_best_move),
            "moves_with_updated_best_move:",
            end=" ",
        )
        for move in self.moves_with_updated_best_move:
            print(move, end=" ")
        print(
            "\n", len(self.moves_with_updated_over), "moves_with_updated_over", end=" "
        )
        for move in self.moves_with_updated_over:
            print(move, end=" ")
        print()

    def empty(self) -> bool:
        """
        Check if all the components of the update instructions block are empty.

        Returns:
            bool: True if all components are empty, False otherwise.
        """
        empty_bool = (
            not bool(self.moves_with_updated_value)
            and not bool(self.moves_with_updated_best_move)
            and not bool(self.moves_with_updated_over)
        )
        return empty_bool
