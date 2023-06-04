import chipiron as ch
import chipiron.players.treevalue.nodes as node
from chipiron.players.treevalue import node_factory as node_fact
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from typing import List
from dataclasses import dataclass


@dataclass
class TreeExpansion:
    """ the class describing TreeExpansion"""

    child_node: node.TreeNode
    parent_node: node.TreeNode | None
    creation_child_node: bool


class TreeExpansions:
    """ the class logging some expansions of a tree"""
    history: List[TreeExpansion]

    def __init__(self):
        self.history = []

    def __iter__(self):
        return iter(self.history)

    def add(self, tree_expansion: TreeExpansion):
        self.history.append(tree_expansion)

class TreeExpansionHistory:
    """ the class logging all the expansions of a tree"""
    history: List[TreeExpansion]

    def __init__(self, root_node: node.TreeNode):
        self.history = [TreeExpansion(child_node=root_node, parent_node=None, creation_child_node=True)]


def create_tree_move(tree: MoveAndValueTree,
                     half_move: int,
                     fast_rep: str,
                     parent_node: node.TreeNode) -> TreeExpansion:
    child_node: node.TreeNode = tree.descendants[half_move][fast_rep]  # add it to the list of descendants
    child_node.add_parent(parent_node)

    tree_expansion: TreeExpansion = TreeExpansion(child_node=child_node,
                                                  parent_node=parent_node,
                                                  creation_child_node=False)
    return tree_expansion


class TreeExpander:
    """
    class in charge of expanding a tree both in terms of adding moves or nodes
    """
    node_factory: node_fact.TreeNodeFactory

    def __init__(self,
                 node_factory: node_fact.TreeNodeFactory,
                 board_evaluator) -> None:
        self.node_factory = node_factory
        self.board_evaluator = board_evaluator

    def create_tree_node(self,
                         tree: MoveAndValueTree,
                         board: ch.chess.BoardChi,
                         board_modifications,
                         half_move: int,
                         parent_node) -> TreeExpansion:
        board_depth: int = half_move - tree.tree_root_half_move
        new_node: node.TreeNode = self.node_factory.create(board=board,
                                                           half_move=half_move,
                                                           count=tree.nodes_count,
                                                           parent_node=parent_node,
                                                           board_depth=board_depth)
        tree.nodes_count += 1
        self.board_evaluator.compute_representation(new_node, parent_node, board_modifications)
        self.board_evaluator.add_evaluation_query(new_node)

        tree_expansion: TreeExpansion = TreeExpansion(child_node=new_node,
                                                      parent_node=parent_node,
                                                      creation_child_node=True)

        return tree_expansion
