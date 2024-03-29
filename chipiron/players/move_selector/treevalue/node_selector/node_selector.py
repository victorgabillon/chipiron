"""
Interface
"""

import typing
from dataclasses import dataclass
from typing import Protocol

from .opening_instructions import OpeningInstructions
from ..trees import MoveAndValueTree

if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.tree_manager as tree_man


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
            latest_tree_expansions: 'tree_man.TreeExpansions'
    ) -> OpeningInstructions:
        ...
