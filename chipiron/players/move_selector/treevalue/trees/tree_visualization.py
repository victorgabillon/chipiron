"""
This module provides functions for visualizing and saving tree structures.

The functions in this module allow for the visualization of tree structures using the Graphviz library.
It provides a way to display the tree structure as a graph and save it as a PDF file.
Additionally, it provides a function to save the raw data of the tree structure to a file using pickle.

Functions:
- add_dot(dot: Digraph, treenode: ITreeNode) -> None: Adds nodes and edges to the graph representation of the tree.
- display_special(node: ITreeNode, format: str, index: dict[chess.Move, str]) -> Digraph: Displays a special
representation of the tree with additional information.
- display(tree: MoveAndValueTree, format_str: str) -> Digraph: Displays the tree structure as a graph.
- save_pdf_to_file(tree: MoveAndValueTree) -> None: Saves the tree structure as a PDF file.
- save_raw_data_to_file(tree: MoveAndValueTree, count: str = '#') -> None: Saves the raw data of the tree
structure to a file.
"""

import pickle
from typing import Any

from graphviz import Digraph

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .move_and_value_tree import MoveAndValueTree


def add_dot(dot: Digraph, treenode: ITreeNode[Any]) -> None:
    """
    Adds a node and edges to the given Dot graph based on the provided tree node.

    Args:
        dot (Digraph): The Dot graph to add the node and edges to.
        treenode (ITreeNode): The tree node to visualize.

    Returns:
        None
    """
    nd = treenode.dot_description()
    dot.node(str(treenode.id), nd)
    move: moveKey
    for _, move in enumerate(treenode.moves_children):
        if treenode.moves_children[move] is not None:
            child = treenode.moves_children[move]
            if child is not None:
                cdd = str(child.id)
                dot.edge(
                    str(treenode.id),
                    cdd,
                    str(treenode.board.get_uci_from_move_key(move_key=move)),
                )
                add_dot(dot, child)


def display_special(
    node: ITreeNode[Any], format_str: str, index: dict[moveKey, str]
) -> Digraph:
    """
    Display a special visualization of a tree node and its children.

    Args:
        node (ITreeNode): The tree node to display.
        format_str (str): The format of the output graph (e.g., 'png', 'pdf', 'svg').
        index (Dict[chess.Move, str]): A dictionary mapping chess moves to their descriptions.

    Returns:
        Digraph: The graph representing the tree visualization.

    Raises:
        AssertionError: If the child node is None or if the parent node is not an AlgorithmNode.
    """
    dot = Digraph(format=format_str)
    print(";;;", type(node))
    nd = node.dot_description()
    dot.node(str(node.id), nd)
    sorted_moves = [(str(move), move) for move in node.moves_children.keys()]
    sorted_moves.sort()
    for move_key in sorted_moves:
        move = move_key[1]
        if node.moves_children[move] is not None:
            child = node.moves_children[move]
            assert child is not None
            assert isinstance(node, AlgorithmNode)

            cdd = str(child.id)
            edge_description = (
                index[move]
                + "|"
                + str(node.board.get_uci_from_move_key(move_key=move))
                + "|"
                + node.minmax_evaluation.description_tree_visualizer_move(child)
            )
            dot.edge(str(node.id), cdd, edge_description)
            dot.node(str(child.id), child.dot_description())
            print("--move:", edge_description)
            print("--child:", child.dot_description())
    return dot


def display(tree: MoveAndValueTree, format_str: str) -> Digraph:
    """
    Display the move and value tree using graph visualization.

    Args:
        tree (MoveAndValueTree): The move and value tree to be displayed.
        format_str (str): The format of the output graph (e.g., 'png', 'pdf', 'svg').

    Returns:
        Digraph: The graph representation of the move and value tree.
    """
    dot = Digraph(format=format_str)
    add_dot(dot, tree.root_node)
    return dot


def save_pdf_to_file(tree: MoveAndValueTree) -> None:
    """
    Saves the visualization of a tree as a PDF file.

    Args:
        tree (MoveAndValueTree): The tree to be visualized and saved.

    Returns:
        None
    """
    dot = display(tree=tree, format_str="pdf")
    round_ = len(tree.root_node.board.move_history_stack) + 2
    color = "white" if tree.root_node.player_to_move else "black"
    dot.render(
        "chipiron/runs/treedisplays/TreeVisual_" + str(int(round_ / 2)) + color + ".pdf"
    )


def save_raw_data_to_file(tree: MoveAndValueTree, count: str = "#") -> None:
    """
    Save raw data of a MoveAndValueTree to a file.

    Args:
        tree (MoveAndValueTree): The MoveAndValueTree object to save.
        count (str, optional): A string to append to the filename. Defaults to '#'.

    Returns:
        None
    """
    round_ = len(tree.root_node.board.move_history_stack) + 2
    color = "white" if tree.root_node.player_to_move else "black"
    filename = (
        "chipiron/debugTreeData_"
        + str(int(round_ / 2))
        + color
        + "-"
        + str(count)
        + ".td"
    )

    import sys

    sys.setrecursionlimit(100000)
    with open(filename, "wb") as f:
        pickle.dump([tree.descendants, tree.root_node], f)
