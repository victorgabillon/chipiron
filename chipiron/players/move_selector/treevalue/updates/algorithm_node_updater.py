import chipiron.players.move_selector.treevalue.nodes as nodes
from .minmax_evaluation_updater import MinMaxEvaluationUpdater
from .updates_file import UpdateInstructions, UpdateInstructionsBatch, ValueUpdateInstructionsBlock
import chipiron.players.move_selector.treevalue.tree_manager as tree_man


class AlgorithmNodeUpdater:
    minmax_evaluation_updater: MinMaxEvaluationUpdater

    def __init__(self,
                 minmax_evaluation_updater: MinMaxEvaluationUpdater
                 ):
        self.minmax_evaluation_updater = minmax_evaluation_updater

    def create_update_instructions_after_node_birth(
            self,
            new_node: nodes.AlgorithmNode
    ) -> UpdateInstructions:
        update_instructions: UpdateInstructions = UpdateInstructions()
        value_update_instructions_block = self.minmax_evaluation_updater.create_update_instructions_after_node_birth(
            new_node=new_node)
        update_instructions.all_instructions_blocks['base'] = value_update_instructions_block
        return update_instructions

    def generate_update_instructions(
            self,
            tree_expansions: tree_man.TreeExpansions
    ) -> UpdateInstructionsBatch:
        # TODO is the way of merging now overkill?

        update_instructions_batch: UpdateInstructionsBatch = UpdateInstructionsBatch()

        tree_expansion: tree_man.TreeExpansion
        for tree_expansion in tree_expansions:
            update_instructions = self.create_update_instructions_after_node_birth(
                new_node=tree_expansion.child_node)
            # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back
            new_update_instructions_batch = UpdateInstructionsBatch({tree_expansion.parent_node: update_instructions})

            # concatenate the update instructions
            update_instructions_batch.merge(new_update_instructions_batch)

        return update_instructions_batch

    def perform_updates(self, node_to_update: nodes.AlgorithmNode,
                        update_instructions: UpdateInstructions) -> UpdateInstructions:
        new_instructions: UpdateInstructions = UpdateInstructions()
        base_update_instructions_block: ValueUpdateInstructionsBlock = self.minmax_evaluation_updater.perform_updates(
            node_to_update,
            updates_instructions=update_instructions)

        new_instructions.all_instructions_blocks['base'] = base_update_instructions_block

        return new_instructions
