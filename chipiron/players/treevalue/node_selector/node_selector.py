"""
Interface
"""

from typing import Protocol
from dataclasses import dataclass

from .opening_instructions import OpeningInstructions
from ..trees import MoveAndValueTree
import chipiron.players.treevalue.tree_manager as tree_man


@dataclass
class NodeSelectorState:
    ...


class NodeSelector(Protocol):
    """
    Protocol for Node Selectors
    """

    def choose_node_and_move_to_open(
            self,
            tree: MoveAndValueTree,
            latest_tree_expansions : tree_man.TreeExpansions
    ) -> OpeningInstructions:
        ...
