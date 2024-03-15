"""
Interface
"""

from dataclasses import dataclass
from typing import Protocol

import chipiron.players.move_selector.treevalue.tree_manager as tree_man
from .opening_instructions import OpeningInstructions
from ..trees import MoveAndValueTree


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
            latest_tree_expansions: tree_man.TreeExpansions
    ) -> OpeningInstructions:
        ...
