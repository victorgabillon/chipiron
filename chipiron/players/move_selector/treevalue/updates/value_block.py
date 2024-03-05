from dataclasses import dataclass, field
import typing

import chipiron.players.move_selector.treevalue.nodes as nodes


@dataclass(slots=True)
class ValueUpdateInstructionsBlock:
    children_with_updated_over: set[nodes.AlgorithmNode] = field(default_factory=set)
    children_with_updated_value: set[nodes.AlgorithmNode] = field(default_factory=set)
    children_with_updated_best_move: set[nodes.AlgorithmNode] = field(default_factory=set)

    def merge(
            self,
            an_update_instruction,
            another_update_instruction
    ) -> None:

        self.children_with_updated_value = an_update_instruction.children_with_updated_value | another_update_instruction.children_with_updated_value
        self.children_with_updated_best_move = an_update_instruction.children_with_updated_best_move | another_update_instruction.children_with_updated_best_move
        self.children_with_updated_over = an_update_instruction.children_with_updated_over | another_update_instruction.children_with_updated_over

    def print_info(self):
        print('upInstructions printing')
        print(len(self.children_with_updated_value), 'children_with_updated_value', end=' ')
        for child in self.children_with_updated_value:
            print(child.id, end=' ')
        print('\n', len(self.children_with_updated_best_move), 'children_with_updated_best_move:', end=' ')
        for child in self.children_with_updated_best_move:
            print(child.id, end=' ')
        print('\n', len(self.children_with_updated_over), 'children_with_updated_over', end=' ')
        for child in self.children_with_updated_over:
            print(child.id, end=' ')
        print()

    def empty(self):
        """ returns if all the components are simultaneously empty"""
        empty_bool = not bool(self.children_with_updated_value) and not bool(
            self.children_with_updated_best_move) and not bool(self.children_with_updated_over)
        return empty_bool


def create_value_update_instructions_block(
        node_sending_update=None,  # node(or None)
        is_node_newly_over=None,  # boolean
        new_value_for_node=None,  # boolean
        new_best_move_for_node=None  # boolean
) -> ValueUpdateInstructionsBlock:
    value_update_instructions_block = ValueUpdateInstructionsBlock(
        children_with_updated_over={node_sending_update} if is_node_newly_over else set(),
        children_with_updated_value={node_sending_update} if new_value_for_node else set(),
        children_with_updated_best_move={node_sending_update} if new_best_move_for_node else set()
    )
    return value_update_instructions_block
