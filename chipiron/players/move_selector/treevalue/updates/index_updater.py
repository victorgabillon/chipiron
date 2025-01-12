"""
This module contains the IndexUpdater class, which is responsible for updating the indices of AlgorithmNode objects in a tree structure.
"""

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import (
    MaxDepthDescendants,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .index_block import (
    IndexUpdateInstructionsFromOneNode,
    IndexUpdateInstructionsTowardsOneParentNode,
)
from .updates_file import UpdateInstructionsTowardsOneParentNode


class IndexUpdater:
    """
    The IndexUpdater class is responsible for updating the indices of AlgorithmNode objects in a tree structure.
    """

    def __init__(self) -> None:
        pass

    def create_update_instructions_after_node_birth(
        self, new_node: AlgorithmNode
    ) -> IndexUpdateInstructionsFromOneNode:
        """
        Creates the update instructions block after a new node is added to the tree.

        Args:
            new_node (AlgorithmNode): The newly added node.

        Returns:
            IndexUpdateInstructionsBlock: The update instructions block.
        """
        base_update_instructions: IndexUpdateInstructionsFromOneNode = (
            IndexUpdateInstructionsFromOneNode(
                node_sending_update=new_node, updated_index=True
            )
        )
        return base_update_instructions

    def perform_updates(
        self,
        node_to_update: AlgorithmNode,
        updates_instructions: UpdateInstructionsTowardsOneParentNode,
    ) -> IndexUpdateInstructionsFromOneNode:
        """
        Performs the index updates based on the given update instructions.

        Args:
            node_to_update (AlgorithmNode): The node to update.
            updates_instructions (UpdateInstructionsTowardsOneParentNode): The update instructions toward this node.

        Returns:
            IndexUpdateInstructionsFromOneNode: The update instructions coming from the updated node.
        """
        # get the base block
        updates_instructions_index: IndexUpdateInstructionsTowardsOneParentNode | None
        updates_instructions_index = (
            updates_instructions.index_updates_toward_one_parent_node
        )
        assert updates_instructions_index is not None

        # UPDATE index
        has_index_changed: bool = False
        move: moveKey
        for move in updates_instructions_index.moves_with_updated_index:
            # hardcoded at some point it should be linked to updater coming from search factory i believe
            assert isinstance(
                node_to_update.exploration_index_data, MaxDepthDescendants
            )
            child = node_to_update.tree_node.moves_children[move]
            assert isinstance(child, AlgorithmNode)
            assert isinstance(child.exploration_index_data, MaxDepthDescendants)
            has_index_changed_child: bool = (
                node_to_update.exploration_index_data.update_from_child(
                    child.exploration_index_data.max_depth_descendants
                )
            )
            has_index_changed = has_index_changed or has_index_changed_child

        base_update_instructions: IndexUpdateInstructionsFromOneNode = (
            IndexUpdateInstructionsFromOneNode(
                node_sending_update=node_to_update, updated_index=has_index_changed
            )
        )

        return base_update_instructions

        # todo i dont understand anymore when the instructions stops beeing propagated back
