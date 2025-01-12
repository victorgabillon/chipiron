"""
This module contains classes for managing update instructions in a batch.

Classes:
- UpdateInstructions: Represents update instructions for a single node.
- UpdateInstructionsBatch: Represents a batch of update instructions for multiple nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Self

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.utils.dict_of_numbered_dict_with_pointer_on_max import (
    DictOfNumberedDictWithPointerOnMax,
)

from .index_block import (
    IndexUpdateInstructionsFromOneNode,
    IndexUpdateInstructionsTowardsOneParentNode,
)
from .value_block import (
    ValueUpdateInstructionsFromOneNode,
    ValueUpdateInstructionsTowardsOneParentNode,
)


@dataclass(slots=True)
class UpdateInstructionsFromOneNode:
    """
    Represents update instructions generated from a single node.

    Attributes:
    - value_block: The value update instructions generated from a single node.
    - index_block: The index update instructions generated from a single node.
    """

    value_block: ValueUpdateInstructionsFromOneNode | None = None
    index_block: IndexUpdateInstructionsFromOneNode | None = None


@dataclass(slots=True)
class UpdateInstructionsTowardsOneParentNode:
    """
    Represents update instructions for a single node.

    Attributes:
    - value_block: The value update instructions block.
    - index_block: The index update instructions block.
    """

    value_updates_toward_one_parent_node: (
        ValueUpdateInstructionsTowardsOneParentNode | None
    ) = None
    index_updates_toward_one_parent_node: (
        IndexUpdateInstructionsTowardsOneParentNode | None
    ) = None

    def add_update_from_a_child_node(
        self,
        update_from_a_child_node: UpdateInstructionsFromOneNode,
        move_from_parent_to_child: moveKey,
    ) -> None:
        assert self.value_updates_toward_one_parent_node is not None
        assert update_from_a_child_node.value_block is not None
        self.value_updates_toward_one_parent_node.add_update_from_one_child_node(
            move_from_parent_to_child=move_from_parent_to_child,
            update_from_one_child_node=update_from_a_child_node.value_block,
        )

        if self.index_updates_toward_one_parent_node is None:
            assert update_from_a_child_node.index_block is None
        else:
            if update_from_a_child_node.index_block is not None:
                self.index_updates_toward_one_parent_node.add_update_from_one_child_node(
                    move_from_parent_to_child=move_from_parent_to_child,
                    update_from_one_child_node=update_from_a_child_node.index_block,
                )

    def add_updates_towards_one_parent_node(self, another_update: Self) -> None:
        assert self.value_updates_toward_one_parent_node is not None
        assert another_update.value_updates_toward_one_parent_node is not None
        self.value_updates_toward_one_parent_node.add_update_toward_one_parent_node(
            another_update.value_updates_toward_one_parent_node
        )

        if self.index_updates_toward_one_parent_node is None:
            assert another_update.index_updates_toward_one_parent_node is None
        else:
            if another_update.index_updates_toward_one_parent_node is not None:
                self.index_updates_toward_one_parent_node.add_update_toward_one_parent_node(
                    another_update.index_updates_toward_one_parent_node
                )

    def print_info(self) -> None:
        """
        Prints information about the update instructions.
        """
        print("printing info of update instructions")
        assert (
            self.index_updates_toward_one_parent_node is not None
            and self.value_updates_toward_one_parent_node is not None
        )
        self.value_updates_toward_one_parent_node.print_info()
        self.index_updates_toward_one_parent_node.print_info()

    def empty(self) -> bool:
        """
        Checks if the update instructions are empty.

        Returns:
        - True if the update instructions are empty, False otherwise.
        """
        assert self.value_updates_toward_one_parent_node is not None
        return self.value_updates_toward_one_parent_node.empty() and (
            self.index_updates_toward_one_parent_node is None
            or self.index_updates_toward_one_parent_node.empty()
        )


@dataclass
class UpdateInstructionsTowardsMultipleNodes:
    # mapping from nodes to the update instructions that are intended to them for consideration (performing the updates)
    one_node_instructions: DictOfNumberedDictWithPointerOnMax[
        ITreeNode[Any], UpdateInstructionsTowardsOneParentNode
    ] = field(default_factory=DictOfNumberedDictWithPointerOnMax)

    def add_update_from_one_child_node(
        self,
        update_from_child_node: UpdateInstructionsFromOneNode,
        parent_node: ITreeNode[Any],
        move_from_parent: moveKey,
    ) -> None:
        if parent_node not in self.one_node_instructions:
            # build the UpdateInstructionsTowardsOneParentNode
            assert update_from_child_node.value_block is not None
            value_updates_toward_one_parent_node: (
                ValueUpdateInstructionsTowardsOneParentNode
            )
            value_updates_toward_one_parent_node = (
                ValueUpdateInstructionsTowardsOneParentNode(
                    moves_with_updated_value=(
                        {move_from_parent}
                        if update_from_child_node.value_block.new_value_for_node
                        else set()
                    ),
                    moves_with_updated_over=(
                        {move_from_parent}
                        if update_from_child_node.value_block.is_node_newly_over
                        else set()
                    ),
                    moves_with_updated_best_move=(
                        {move_from_parent}
                        if update_from_child_node.value_block.new_best_move_for_node
                        else set()
                    ),
                )
            )
            index_updates_toward_one_parent_node: (
                IndexUpdateInstructionsTowardsOneParentNode | None
            )
            if update_from_child_node.index_block is not None:
                index_updates_toward_one_parent_node = (
                    IndexUpdateInstructionsTowardsOneParentNode(
                        moves_with_updated_index=(
                            {move_from_parent}
                            if update_from_child_node.index_block.updated_index
                            else set()
                        ),
                    )
                )
            else:
                index_updates_toward_one_parent_node = None
            update_instructions_towards_parent: UpdateInstructionsTowardsOneParentNode
            update_instructions_towards_parent = UpdateInstructionsTowardsOneParentNode(
                value_updates_toward_one_parent_node=value_updates_toward_one_parent_node,
                index_updates_toward_one_parent_node=index_updates_toward_one_parent_node,
            )
            self.one_node_instructions[parent_node] = update_instructions_towards_parent
        else:
            # update the UpdateInstructionsTowardsOneParentNode
            self.one_node_instructions[parent_node].add_update_from_a_child_node(
                update_from_a_child_node=update_from_child_node,
                move_from_parent_to_child=move_from_parent,
            )

    def add_updates_towards_one_parent_node(
        self,
        update_from_child_node: UpdateInstructionsTowardsOneParentNode,
        parent_node: ITreeNode[Any],
    ) -> None:
        if parent_node in self.one_node_instructions:
            self.one_node_instructions[parent_node].add_updates_towards_one_parent_node(
                another_update=update_from_child_node
            )
        else:
            self.one_node_instructions[parent_node] = update_from_child_node

    def pop_item(self) -> tuple[ITreeNode[Any], UpdateInstructionsTowardsOneParentNode]:
        return self.one_node_instructions.popitem()

    def __bool__(self) -> bool:
        """
        Checks if the data structure is non-empty.

        Returns:
            bool: True if the data structure is non-empty, False otherwise.
        """
        return bool(self.one_node_instructions)
