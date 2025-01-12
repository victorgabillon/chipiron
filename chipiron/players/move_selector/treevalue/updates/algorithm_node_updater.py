"""
This module contains the AlgorithmNodeUpdater class, which is responsible for updating AlgorithmNode objects in a
 tree structure.

The AlgorithmNodeUpdater class provides methods for creating update instructions after a node is added to the
 tree, generating update instructions for a batch of tree expansions, and performing updates on a specific node
  based on the given update instructions.
"""

import typing
from dataclasses import dataclass

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .index_block import IndexUpdateInstructionsFromOneNode
from .index_updater import IndexUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater
from .updates_file import (
    UpdateInstructionsFromOneNode,
    UpdateInstructionsTowardsMultipleNodes,
    UpdateInstructionsTowardsOneParentNode,
)
from .value_block import ValueUpdateInstructionsFromOneNode

if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.tree_manager as tree_man


@dataclass
class AlgorithmNodeUpdater:
    """
    The AlgorithmNodeUpdater class is responsible for updating AlgorithmNode objects in a tree.

    Attributes:
        minmax_evaluation_updater (MinMaxEvaluationUpdater): The updater for min-max evaluation values.
        index_updater (IndexUpdater | None): The updater for node indices, if available.
    """

    minmax_evaluation_updater: MinMaxEvaluationUpdater
    index_updater: IndexUpdater | None = None

    def create_update_instructions_after_node_birth(
        self, new_node: AlgorithmNode
    ) -> UpdateInstructionsFromOneNode:
        """
        Creates update instructions after a new node is added to the tree.

        Args:
            new_node (AlgorithmNode): The newly added AlgorithmNode.

        Returns:
            UpdateInstructions: The update instructions for the new node.
        """
        value_update_instructions: ValueUpdateInstructionsFromOneNode
        value_update_instructions = (
            self.minmax_evaluation_updater.create_update_instructions_after_node_birth(
                new_node=new_node
            )
        )

        index_update_instructions: IndexUpdateInstructionsFromOneNode | None
        if self.index_updater is not None:
            index_update_instructions = (
                self.index_updater.create_update_instructions_after_node_birth(
                    new_node=new_node
                )
            )
        else:
            index_update_instructions = None

        update_instructions: UpdateInstructionsFromOneNode = (
            UpdateInstructionsFromOneNode(
                value_block=value_update_instructions,
                index_block=index_update_instructions,
            )
        )

        return update_instructions

    def generate_update_instructions(
        self, tree_expansions: "tree_man.TreeExpansions"
    ) -> UpdateInstructionsTowardsMultipleNodes:
        """
        Generates update instructions for a batch of tree expansions.

        Args:
            tree_expansions (tree_man.TreeExpansions): The batch of tree expansions.

        Returns:
            UpdateInstructionsBatch: The update instructions for the batch of tree expansions.
        """
        # TODO is the way of merging now overkill?

        update_instructions_batch: UpdateInstructionsTowardsMultipleNodes = (
            UpdateInstructionsTowardsMultipleNodes()
        )

        tree_expansion: "tree_man.TreeExpansion"
        for tree_expansion in tree_expansions:
            assert isinstance(tree_expansion.child_node, AlgorithmNode)
            update_instructions: UpdateInstructionsFromOneNode = (
                self.create_update_instructions_after_node_birth(
                    new_node=tree_expansion.child_node
                )
            )
            # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back

            assert tree_expansion.parent_node is not None
            # looks like we should not update from the root node backward!

            assert tree_expansion.move is not None
            update_instructions_batch.add_update_from_one_child_node(
                update_from_child_node=update_instructions,
                parent_node=tree_expansion.parent_node,
                move_from_parent=tree_expansion.move,
            )

        return update_instructions_batch

    def perform_updates(
        self,
        node_to_update: AlgorithmNode,
        update_instructions: UpdateInstructionsTowardsOneParentNode,
    ) -> UpdateInstructionsFromOneNode:
        """
        Performs updates on a specific node based on the given update instructions.

        Args:
            node_to_update (AlgorithmNode): The node to update.
            update_instructions (UpdateInstructions): The update instructions for the node.

        Returns:
            UpdateInstructions: The new update instructions after performing the updates.
        """
        value_update_instructions: ValueUpdateInstructionsFromOneNode = (
            self.minmax_evaluation_updater.perform_updates(
                node_to_update, updates_instructions=update_instructions
            )
        )

        index_update_instructions: IndexUpdateInstructionsFromOneNode | None
        if self.index_updater is not None:
            index_update_instructions = self.index_updater.perform_updates(
                node_to_update, updates_instructions=update_instructions
            )
        else:
            index_update_instructions = None

        new_update_instructions: UpdateInstructionsFromOneNode = (
            UpdateInstructionsFromOneNode(
                value_block=value_update_instructions,
                index_block=index_update_instructions,
            )
        )

        return new_update_instructions
