import chipiron.players.treevalue.nodes as node
from typing import List
from dataclasses import dataclass
import chipiron.chessenvironment.board as board_mod


@dataclass
class TreeExpansion:
    """ the class describing TreeExpansion"""

    child_node: node.ITreeNode
    parent_node: node.ITreeNode | None
    board_modifications: board_mod.BoardModification| None
    creation_child_node: bool


class TreeExpansions:
    """ the class logging some expansions of a tree"""
    history: List[TreeExpansion]

    def __init__(self):
        self.expansions_with_node_creation = []
        self.expansions_without_node_creation = []

    def __iter__(self):
        return iter(self.expansions_with_node_creation + self.expansions_without_node_creation)

    def add(self, tree_expansion: TreeExpansion):
        if tree_expansion.creation_child_node:
            self.add_creation(tree_expansion=tree_expansion)
        else:
            self.add_connection(tree_expansion=tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion):
        self.expansions_with_node_creation.append(tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion):
        self.expansions_without_node_creation.append(tree_expansion)


class TreeExpansionHistory:
    """ the class logging all the expansions of a tree"""
    history: List[TreeExpansion]

    def __init__(self, root_node: node.TreeNode):
        self.history = [TreeExpansion(child_node=root_node, parent_node=None, creation_child_node=True)]




