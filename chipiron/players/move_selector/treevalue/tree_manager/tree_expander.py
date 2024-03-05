import chipiron.players.move_selector.treevalue.nodes as node
from typing import List
from dataclasses import dataclass, field
import chipiron.environments.chess.board as board_mod


@dataclass(slots=True)
class TreeExpansion:
    """ the class describing TreeExpansion"""

    child_node: node.ITreeNode
    parent_node: node.ITreeNode | None
    board_modifications: board_mod.BoardModification | None
    creation_child_node: bool

    def __repr__(self):
        return (f'child_node{self.child_node.id} | '
                f'parent_node{self.parent_node.id if self.parent_node is not None else None} | '
                f'creation_child_node{self.creation_child_node}')


@dataclass(slots=True)
class TreeExpansions:
    """ the class logging some expansions of a tree"""

    expansions_with_node_creation: List[TreeExpansion] = field(default_factory=list)
    expansions_without_node_creation: List[TreeExpansion] = field(default_factory=list)

    def __iter__(self):
        return iter(self.expansions_with_node_creation + self.expansions_without_node_creation)

    def add(
            self,
            tree_expansion: TreeExpansion
    ) -> None:
        if tree_expansion.creation_child_node:
            self.add_creation(tree_expansion=tree_expansion)
        else:
            self.add_connection(tree_expansion=tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion):
        self.expansions_with_node_creation.append(tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion):
        self.expansions_without_node_creation.append(tree_expansion)

    def __str__(self):
        return (f'expansions_with_node_creation {self.expansions_with_node_creation} \n'
                f'expansions_without_node_creation{self.expansions_without_node_creation}')


class TreeExpansionHistory:
    """ the class logging all the expansions of a tree"""
    history: List[TreeExpansion]

    def __init__(self, root_node: node.TreeNode):
        self.history = [TreeExpansion(
            child_node=root_node,
            parent_node=None,
            creation_child_node=True,
            board_modifications=None
        )]
