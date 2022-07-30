from typing import Protocol
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructionsBatch
from chipiron.players.treevalue.trees import MoveAndValueTree


class NodeSelector(Protocol):

    def choose_node_and_move_to_open(self, tree: MoveAndValueTree) -> OpeningInstructionsBatch:
        ...

    def update_after_node_creation(self, node, parent_node) -> None:
        ...
