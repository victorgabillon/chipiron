from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from .updates_file import UpdateInstructions
from .value_block import ValueUpdateInstructionsBlock, create_value_update_instructions_block


class MinMaxEvaluationUpdater:

    def __init__(self):
        pass

    def create_update_instructions_after_node_birth(
            self,
            new_node: AlgorithmNode
    ) -> ValueUpdateInstructionsBlock:

        base_update_instructions_block: ValueUpdateInstructionsBlock
        base_update_instructions_block = create_value_update_instructions_block(
            node_sending_update=new_node,
            is_node_newly_over=new_node.minmax_evaluation.over_event.is_over(),
            new_value_for_node=True,
            new_best_move_for_node=False
        )
        return base_update_instructions_block

    def perform_updates(
            self,
            node_to_update: AlgorithmNode,
            updates_instructions: UpdateInstructions
    ) -> ValueUpdateInstructionsBlock:
        # get the base block
        updates_instructions_block: ValueUpdateInstructionsBlock | None = updates_instructions.value_block
        assert updates_instructions_block is not None

        # UPDATE VALUE
        has_value_changed, has_best_node_seq_changed_1 = node_to_update.minmax_evaluation.minmax_value_update_from_children(
            children_with_updated_value=updates_instructions_block.children_with_updated_value
        )

        # UPDATE BEST MOVE
        if updates_instructions_block.children_with_updated_best_move:
            has_best_node_seq_changed_2 = node_to_update.minmax_evaluation.update_best_move_sequence(
                updates_instructions_block.children_with_updated_best_move)
        else:
            has_best_node_seq_changed_2 = False
        has_best_node_seq_changed = has_best_node_seq_changed_1 or has_best_node_seq_changed_2

        # UPDATE OVER
        is_newly_over = node_to_update.minmax_evaluation.update_over(
            updates_instructions_block.children_with_updated_over)

        assert (is_newly_over is not None)

        # create the new instructions for the parents
        base_update_instructions_block: ValueUpdateInstructionsBlock
        base_update_instructions_block = create_value_update_instructions_block(
            node_sending_update=node_to_update,
            is_node_newly_over=is_newly_over,
            new_value_for_node=has_value_changed,
            new_best_move_for_node=has_best_node_seq_changed
        )

        return base_update_instructions_block
