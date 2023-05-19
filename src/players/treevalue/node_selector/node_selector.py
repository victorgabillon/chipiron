from typing import Protocol
from src.players.treevalue.trees.opening_instructions import OpeningInstructionsBatch


class NodeSelector(Protocol):

    def choose_node_and_move_to_open(self, tree) -> OpeningInstructionsBatch:
        ...

    def update_after_node_creation(self, node, parent_node) -> None:
        ...