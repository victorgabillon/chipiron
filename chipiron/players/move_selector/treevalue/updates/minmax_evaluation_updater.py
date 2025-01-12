"""
This module contains the MinMaxEvaluationUpdater class, which is responsible for updating the min-max evaluation values of AlgorithmNode objects.
"""

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .updates_file import UpdateInstructionsTowardsOneParentNode
from .value_block import (
    ValueUpdateInstructionsFromOneNode,
    ValueUpdateInstructionsTowardsOneParentNode,
)


class MinMaxEvaluationUpdater:
    """
    The MinMaxEvaluationUpdater class is responsible for updating the min-max evaluation values of AlgorithmNode objects.
    """

    def __init__(self) -> None:
        """Initializes a new instance of the MinMaxEvaluationUpdater class."""
        pass

    def create_update_instructions_after_node_birth(
        self, new_node: AlgorithmNode
    ) -> ValueUpdateInstructionsFromOneNode:
        """
        Creates the update instructions for a newly created AlgorithmNode.

        Args:
            new_node (AlgorithmNode): The newly created AlgorithmNode.

        Returns:
            ValueUpdateInstructionsBlock: The update instructions for the newly created node.
        """
        base_update_instructions: ValueUpdateInstructionsFromOneNode = (
            ValueUpdateInstructionsFromOneNode(
                node_sending_update=new_node,
                is_node_newly_over=new_node.minmax_evaluation.over_event.is_over(),
                new_value_for_node=True,
                new_best_move_for_node=False,
            )
        )
        return base_update_instructions

    def perform_updates(
        self,
        node_to_update: AlgorithmNode,
        updates_instructions: UpdateInstructionsTowardsOneParentNode,
    ) -> ValueUpdateInstructionsFromOneNode:
        """
        Performs the updates on an AlgorithmNode based on the given update instructions.

        Args:
            node_to_update (AlgorithmNode): The AlgorithmNode to update.
            updates_instructions (UpdateInstructionsTowardsOneParentNode): The update instructions.

        Returns:
            ValueUpdateInstructionsFromOneNode: The update instructions for the parents of the updated node.
        """
        # get the base block
        updates_instructions_block: ValueUpdateInstructionsTowardsOneParentNode | None
        updates_instructions_block = (
            updates_instructions.value_updates_toward_one_parent_node
        )
        assert updates_instructions_block is not None

        # UPDATE VALUE
        has_value_changed: bool
        has_best_node_seq_changed_1: bool
        has_value_changed, has_best_node_seq_changed_1 = (
            node_to_update.minmax_evaluation.minmax_value_update_from_children(
                moves_with_updated_value=updates_instructions_block.moves_with_updated_value
            )
        )

        # UPDATE BEST MOVE
        has_best_node_seq_changed_2: bool
        if updates_instructions_block.moves_with_updated_best_move:
            has_best_node_seq_changed_2 = (
                node_to_update.minmax_evaluation.update_best_move_sequence(
                    updates_instructions_block.moves_with_updated_best_move
                )
            )
        else:
            has_best_node_seq_changed_2 = False
        has_best_node_seq_changed: bool = (
            has_best_node_seq_changed_1 or has_best_node_seq_changed_2
        )

        # UPDATE OVER
        is_newly_over = node_to_update.minmax_evaluation.update_over(
            updates_instructions_block.moves_with_updated_over
        )

        assert is_newly_over is not None

        # create the new instructions for the parents
        base_update_instructions_block: ValueUpdateInstructionsFromOneNode
        base_update_instructions_block = ValueUpdateInstructionsFromOneNode(
            node_sending_update=node_to_update,
            is_node_newly_over=is_newly_over,
            new_value_for_node=has_value_changed,
            new_best_move_for_node=has_best_node_seq_changed,
        )

        return base_update_instructions_block
