from dataclasses import dataclass, field
from typing import Self

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode


@dataclass(slots=True)
class IndexUpdateInstructionsBlock:
    children_with_updated_index: set[AlgorithmNode] = field(default_factory=set)

    def merge(
            self,
            an_update_instruction: Self,
            another_update_instruction: Self
    ) -> None:
        self.children_with_updated_index = an_update_instruction.children_with_updated_index | another_update_instruction.children_with_updated_index

    def print_info(self) -> None:
        print('upInstructions printing')
        print(len(self.children_with_updated_index), 'children_with_updated_index', end=' ')
        for child in self.children_with_updated_index:
            print(child.id, end=' ')
        print()

    def empty(self) -> bool:
        """ returns if all the components are simultaneously empty"""
        empty_bool = not bool(self.children_with_updated_index)
        return empty_bool
