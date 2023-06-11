import chipiron.players.treevalue.nodes as nodes
from .updates_file import ValueUpdateInstructionsBlock, UpdateInstructions


class MinMaxEvaluationUpdater:

    def __init__(self):
        pass

    def create_update_instructions_after_node_birth(self,
                                                    new_node: nodes.AlgorithmNode) -> ValueUpdateInstructionsBlock:
        base_update_instructions_block: ValueUpdateInstructionsBlock
        base_update_instructions_block = ValueUpdateInstructionsBlock(
            node_sending_update=new_node,
            is_node_newly_over=new_node.minmax_evaluation.over_event.is_over(),
            new_value_for_node=True,
            new_best_move_for_node=False
        )
        return base_update_instructions_block

    def perform_updates(self,
                        node_to_update: nodes.AlgorithmNode,
                        updates_instructions: UpdateInstructions) -> ValueUpdateInstructionsBlock:
        # get the base block
        updates_instructions_block: ValueUpdateInstructionsBlock = updates_instructions[
            'base']  # todo create a variable for the tag

        # UPDATE VALUE
        has_value_changed, has_best_node_seq_changed_1 = node_to_update.minmax_evaluation.minmax_value_update_from_children(
            updates_instructions_block['children_with_updated_value'])

        # UPDATE BEST MOVE
        has_best_node_seq_changed_2 = node_to_update.minmax_evaluation.update_best_move_sequence(
            updates_instructions_block['children_with_updated_best_move'])
        has_best_node_seq_changed = has_best_node_seq_changed_1 or has_best_node_seq_changed_2

        # UPDATE OVER
        is_newly_over = node_to_update.minmax_evaluation.update_over(
            updates_instructions_block['children_with_updated_over'])
        assert (is_newly_over is not None)

        # create the new instructions for the parents
        base_update_instructions_block: ValueUpdateInstructionsBlock
        base_update_instructions_block = ValueUpdateInstructionsBlock(node_sending_update=node_to_update,
                                                                      is_node_newly_over=is_newly_over,
                                                                      new_value_for_node=has_value_changed,
                                                                      new_best_move_for_node=has_best_node_seq_changed)

        #        self.test()

        return base_update_instructions_block

        # todo i dont understand anymore when the instructions stops beeing propagated back
