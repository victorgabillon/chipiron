"""
This module contains utility functions for working with tree nodes in the move selector.

Functions:
- are_all_moves_and_children_opened(tree_node: TreeNode) -> bool: Checks if all moves and children of a tree node are opened.
- a_move_sequence_from_root(tree_node: ITreeNode) -> list[str]: Returns a list of move sequences from the root node to a given tree node.
- print_a_move_sequence_from_root(tree_node: TreeNode) -> None: Prints the move sequence from the root node to a given tree node.
- is_winning(node_minmax_evaluation: NodeMinmaxEvaluation, color: chess.Color) -> bool: Checks if the color to play in the node is winning.
"""

import chess

from chipiron.players.move_selector.treevalue.nodes.algorithm_node.node_minmax_evaluation import NodeMinmaxEvaluation
from .itree_node import ITreeNode
from .tree_node import TreeNode


def are_all_moves_and_children_opened(tree_node: TreeNode) -> bool:
    """
    Checks if all moves and children of a tree node are opened.

    Args:
        tree_node (TreeNode): The tree node to check.

    Returns:
        bool: True if all moves and children are opened, False otherwise.
    """
    return tree_node.all_legal_moves_generated and tree_node.non_opened_legal_moves == set()


def a_move_sequence_from_root(tree_node: ITreeNode) -> list[str]:
    """
    Returns a list of move sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the move sequence for.

    Returns:
        list[str]: A list of move sequences from the root node to the given tree node.
    """
    move_sequence_from_root: list[chess.Move] = []
    child: ITreeNode = tree_node
    while child.parent_nodes:
        parent: ITreeNode = next(iter(child.parent_nodes))
        move_sequence_from_root.append(parent.moves_children.inverse[child])
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def print_a_move_sequence_from_root(tree_node: TreeNode) -> None:
    """
    Prints the move sequence from the root node to a given tree node.

    Args:
        tree_node (TreeNode): The tree node to print the move sequence for.

    Returns:
        None
    """
    move_sequence_from_root: list[str] = a_move_sequence_from_root(tree_node=tree_node)
    print(f'a_move_sequence_from_root{move_sequence_from_root}')


def is_winning(node_minmax_evaluation: NodeMinmaxEvaluation, color: chess.Color) -> bool:
    """
    Checks if the color to play in the node is winning.

    Args:
        node_minmax_evaluation (NodeMinmaxEvaluation): The evaluation of the node.
        color (chess.Color): The color to check.

    Returns:
        bool: True if the color is winning, False otherwise.
    """
    assert node_minmax_evaluation.value_white_minmax is not None
    winning_if_color_white: bool = node_minmax_evaluation.value_white_minmax > .98 and color
    winning_if_color_black: bool = node_minmax_evaluation.value_white_minmax < -.98 and not color

    return winning_if_color_white or winning_if_color_black
