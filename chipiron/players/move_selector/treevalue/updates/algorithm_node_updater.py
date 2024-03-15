from dataclasses import dataclass

import chipiron.players.move_selector.treevalue.nodes as nodes
import chipiron.players.move_selector.treevalue.tree_manager as tree_man
from .index_block import IndexUpdateInstructionsBlock
from .index_updater import IndexUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater
from .updates_file import UpdateInstructions, UpdateInstructionsBatch
from .value_block import ValueUpdateInstructionsBlock


@dataclass
class AlgorithmNodeUpdater:
    minmax_evaluation_updater: MinMaxEvaluationUpdater
    index_updater: IndexUpdater | None = None

    def create_update_instructions_after_node_birth(
            self,
            new_node: nodes.AlgorithmNode
    ) -> UpdateInstructions:
        value_update_instructions_block = self.minmax_evaluation_updater.create_update_instructions_after_node_birth(
            new_node=new_node
        )
        if self.index_updater is not None:
            index_update_instructions_block = self.index_updater.create_update_instructions_after_node_birth(
                new_node=new_node
            )
        else:
            index_update_instructions_block = None

        update_instructions: UpdateInstructions = UpdateInstructions(
            value_block=value_update_instructions_block,
            index_block=index_update_instructions_block
        )

        return update_instructions

    def generate_update_instructions(
            self,
            tree_expansions: tree_man.TreeExpansions
    ) -> UpdateInstructionsBatch:
        # TODO is the way of merging now overkill?

        update_instructions_batch: UpdateInstructionsBatch = UpdateInstructionsBatch()

        tree_expansion: tree_man.TreeExpansion
        for tree_expansion in tree_expansions:
            assert isinstance(tree_expansion.child_node, nodes.AlgorithmNode)
            update_instructions = self.create_update_instructions_after_node_birth(
                new_node=tree_expansion.child_node)
            # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back

            assert (tree_expansion.parent_node is not None)
            # looks like we should not update from the root node backward!

            new_update_instructions_batch = UpdateInstructionsBatch({tree_expansion.parent_node: update_instructions})

            # concatenate the update instructions
            update_instructions_batch.merge(new_update_instructions_batch)

        return update_instructions_batch

    def perform_updates(
            self,
            node_to_update: nodes.AlgorithmNode,
            update_instructions: UpdateInstructions
    ) -> UpdateInstructions:

        value_update_instructions_block: ValueUpdateInstructionsBlock = self.minmax_evaluation_updater.perform_updates(
            node_to_update,
            updates_instructions=update_instructions
        )

        index_update_instructions_block: IndexUpdateInstructionsBlock | None
        if self.index_updater is not None:
            index_update_instructions_block = self.index_updater.perform_updates(
                node_to_update,
                updates_instructions=update_instructions
            )
        else:
            index_update_instructions_block = None

        new_update_instructions: UpdateInstructions = UpdateInstructions(
            value_block=value_update_instructions_block,
            index_block=index_update_instructions_block
        )

        return new_update_instructions
