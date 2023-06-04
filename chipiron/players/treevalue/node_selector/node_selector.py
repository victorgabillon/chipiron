from typing import Protocol
from dataclasses import dataclass

from .opening_instructions import OpeningInstructions
from ..trees import MoveAndValueTree


@dataclass
class NodeSelectorState:
    ...


class NodeSelector(Protocol):

    def choose_node_and_move_to_open(self,
                                     tree: MoveAndValueTree,
                                     node_selector_state: NodeSelectorState
                                     ) -> OpeningInstructions:
        ...

    def Communi(self,
                                     tree: MoveAndValueTree,
                                     node_selector_state: NodeSelectorState
                                     ) -> OpeningInstructions:
        ...
