"""
This module contains utility functions for working with tree nodes in the move selector.

Functions:
- are_all_moves_and_children_opened(tree_node: TreeNode) -> bool: Checks if all moves and children of a tree node are opened.
- a_move_sequence_from_root(tree_node: ITreeNode) -> list[str]: Returns a list of move sequences from the root node to a given tree node.
- print_a_move_sequence_from_root(tree_node: TreeNode) -> None: Prints the move sequence from the root node to a given tree node.
- is_winning(node_minmax_evaluation: NodeMinmaxEvaluation, color: chess.Color) -> bool: Checks if the color to play in the node is winning.
"""

from typing import Any

import chess

from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)

from .itree_node import ITreeNode
from .tree_node import TreeNode


def are_all_moves_and_children_opened(tree_node: TreeNode[Any]) -> bool:
    """
    Checks if all moves and children of a tree node are opened.

    Args:
        tree_node (TreeNode): The tree node to check.

    Returns:
        bool: True if all moves and children are opened, False otherwise.
    """
    return (
        tree_node.all_legal_moves_generated
        and tree_node.non_opened_legal_moves == set()
    )


def a_move_key_sequence_from_root(tree_node: ITreeNode[Any]) -> list[str]:
    """
    Returns a list of move sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the move sequence for.

    Returns:
        list[str]: A list of move sequences from the root node to the given tree node.
    """
    move_sequence_from_root: list[moveKey] = []
    child: ITreeNode[Any] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[Any] = next(iter(child.parent_nodes))
        move: moveKey = child.parent_nodes[parent]
        move_sequence_from_root.append(move)
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def a_move_uci_sequence_from_root(tree_node: ITreeNode[Any]) -> list[str]:
    """
    Returns a list of move sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the move sequence for.

    Returns:
        list[str]: A list of move sequences from the root node to the given tree node.
    """
    move_sequence_from_root: list[moveUci] = []
    child: ITreeNode[Any] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[Any] = next(iter(child.parent_nodes))
        move: moveKey = child.parent_nodes[parent]
        move_uci: moveUci = parent.board.get_uci_from_move_key(move)
        move_sequence_from_root.append(move_uci)
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def best_node_sequence_from_node(
    tree_node: AlgorithmNode,
) -> list[ITreeNode[Any]]:
    """ """

    best_move_seq: list[moveKey] = tree_node.minmax_evaluation.best_move_sequence
    index = 0
    move_sequence: list[ITreeNode[Any]] = [tree_node]
    child: ITreeNode[Any] = tree_node
    while child.moves_children:
        move: moveKey = best_move_seq[index]
        child_ = child.moves_children[move]
        assert child_ is not None
        child = child_
        move_sequence.append(child)
        index = index + 1
    return move_sequence


def print_a_move_sequence_from_root(tree_node: TreeNode[Any]) -> None:
    """
    Prints the move sequence from the root node to a given tree node.

    Args:
        tree_node (TreeNode): The tree node to print the move sequence for.

    Returns:
        None
    """
    move_sequence_from_root: list[str] = a_move_key_sequence_from_root(
        tree_node=tree_node
    )
    print(f"a_move_sequence_from_root{move_sequence_from_root}")


def is_winning(
    node_minmax_evaluation: NodeMinmaxEvaluation, color: chess.Color
) -> bool:
    """
    Checks if the color to play in the node is winning.

    Args:
        node_minmax_evaluation (NodeMinmaxEvaluation): The evaluation of the node.
        color (chess.Color): The color to check.

    Returns:
        bool: True if the color is winning, False otherwise.
    """
    assert node_minmax_evaluation.value_white_minmax is not None
    winning_if_color_white: bool = (
        node_minmax_evaluation.value_white_minmax > 0.98 and color
    )
    winning_if_color_black: bool = (
        node_minmax_evaluation.value_white_minmax < -0.98 and not color
    )

    return winning_if_color_white or winning_if_color_black
