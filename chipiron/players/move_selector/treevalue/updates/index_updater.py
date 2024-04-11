"""
This module contains the IndexUpdater class, which is responsible for updating the indices of AlgorithmNode objects in a tree structure.
"""

from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import MaxDepthDescendants
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from .index_block import IndexUpdateInstructionsBlock
from .updates_file import UpdateInstructions


class IndexUpdater:
    """
    The IndexUpdater class is responsible for updating the indices of AlgorithmNode objects in a tree structure.
    """

    def __init__(self) -> None:
        pass

    def create_update_instructions_after_node_birth(
            self,
            new_node: AlgorithmNode
    ) -> IndexUpdateInstructionsBlock:
        """
        Creates the update instructions block after a new node is added to the tree.

        Args:
            new_node (AlgorithmNode): The newly added node.

        Returns:
            IndexUpdateInstructionsBlock: The update instructions block.
        """
        base_update_instructions_block: IndexUpdateInstructionsBlock = IndexUpdateInstructionsBlock(
            children_with_updated_index={new_node}
        )
        return base_update_instructions_block

    def perform_updates(
            self,
            node_to_update: AlgorithmNode,
            updates_instructions: UpdateInstructions
    ) -> IndexUpdateInstructionsBlock:
        """
        Performs the index updates based on the given update instructions.

        Args:
            node_to_update (AlgorithmNode): The node to update.
            updates_instructions (UpdateInstructions): The update instructions.

        Returns:
            IndexUpdateInstructionsBlock: The update instructions block.
        """
        # get the base block
        updates_instructions_block: IndexUpdateInstructionsBlock | None = updates_instructions.index_block
        assert updates_instructions_block is not None

        # UPDATE index
        has_index_changed: bool = False
        for child in updates_instructions_block.children_with_updated_index:
            # hardcoded at some point it should be linked to updater coming from search factory i believe
            assert isinstance(node_to_update.exploration_index_data, MaxDepthDescendants)
            assert isinstance(child.exploration_index_data, MaxDepthDescendants)
            has_index_changed_child: bool = node_to_update.exploration_index_data.update_from_child(
                child.exploration_index_data.max_depth_descendants)
            has_index_changed = has_index_changed or has_index_changed_child

        base_update_instructions_block: IndexUpdateInstructionsBlock = IndexUpdateInstructionsBlock(
            children_with_updated_index={node_to_update} if has_index_changed else set()
        )

        return base_update_instructions_block

        # todo i dont understand anymore when the instructions stops beeing propagated back
