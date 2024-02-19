import chipiron.players.move_selector.treevalue.nodes as nodes
from .updates_file import UpdateInstructions
from .index_block import IndexUpdateInstructionsBlock


class IndexUpdater:

    def __init__(self):
        pass

    def create_update_instructions_after_node_birth(
            self,
            new_node: nodes.AlgorithmNode
    ) -> IndexUpdateInstructionsBlock:
        base_update_instructions_block: IndexUpdateInstructionsBlock = IndexUpdateInstructionsBlock(
            children_with_updated_index={new_node}
        )
        return base_update_instructions_block

    def perform_updates(
            self,
            node_to_update: nodes.AlgorithmNode,
            updates_instructions: UpdateInstructions
    ) -> IndexUpdateInstructionsBlock:
        # get the base block
        updates_instructions_block: IndexUpdateInstructionsBlock = updates_instructions.index_block

        # UPDATE index
        has_index_changed: bool = False
        for child in updates_instructions_block.children_with_updated_index:
            # hardcoded at some point it should be linked to updater coming from search factory i believe
            has_index_changed_child: bool = node_to_update.exploration_index_data.update_from_child(
                child.exploration_index_data.max_depth_descendants)
            has_index_changed = has_index_changed or has_index_changed_child

        base_update_instructions_block: IndexUpdateInstructionsBlock = IndexUpdateInstructionsBlock(
            children_with_updated_index={node_to_update} if has_index_changed else {}
        )

        return base_update_instructions_block

        # todo i dont understand anymore when the instructions stops beeing propagated back
