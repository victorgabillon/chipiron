"""
This module contains classes related to tree expansion in a chess game.
"""

import typing
from dataclasses import dataclass, field
from typing import Any, List

import chipiron.environments.chess.board as board_mod
import chipiron.players.move_selector.treevalue.nodes as node
from chipiron.environments.chess.move.imove import moveKey


@dataclass(slots=True)
class TreeExpansion:
    """
    Represents an expansion of a tree in a chess game.

    Attributes:
        child_node (node.ITreeNode): The child node created during the expansion.
        parent_node (node.ITreeNode | None): The parent node of the child node. None if it's the root node.
        board_modifications (board_mod.BoardModification | None): The modifications made to the chess board during the expansion.
        creation_child_node (bool): Indicates whether the child node was created during the expansion.
        move (chess.Move): the move from parent to child node.
    """

    child_node: node.ITreeNode[Any]
    parent_node: node.ITreeNode[Any] | None
    board_modifications: board_mod.BoardModificationP | None
    creation_child_node: bool
    move: moveKey | None

    def __repr__(self) -> str:
        return (
            f"child_node{self.child_node.id} | "
            f"parent_node{self.parent_node.id if self.parent_node is not None else None} | "
            f"creation_child_node{self.creation_child_node}"
        )


@dataclass(slots=True)
class TreeExpansions:
    """
    Represents a collection of tree expansions in a chess game.

    Attributes:
        expansions_with_node_creation (List[TreeExpansion]): List of expansions where child nodes were created.
        expansions_without_node_creation (List[TreeExpansion]): List of expansions where child nodes were not created.
    """

    expansions_with_node_creation: List[TreeExpansion] = field(default_factory=list)
    expansions_without_node_creation: List[TreeExpansion] = field(default_factory=list)

    def __iter__(self) -> typing.Iterator[TreeExpansion]:
        return iter(
            self.expansions_with_node_creation + self.expansions_without_node_creation
        )

    def add(self, tree_expansion: TreeExpansion) -> None:
        """
        Adds a tree expansion to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        if tree_expansion.creation_child_node:
            self.add_creation(tree_expansion=tree_expansion)
        else:
            self.add_connection(tree_expansion=tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion) -> None:
        """
        Adds a tree expansion with a created child node to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        self.expansions_with_node_creation.append(tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion) -> None:
        """
        Adds a tree expansion without a created child node to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        self.expansions_without_node_creation.append(tree_expansion)

    def __str__(self) -> str:
        return (
            f"expansions_with_node_creation {self.expansions_with_node_creation} \n"
            f"expansions_without_node_creation{self.expansions_without_node_creation}"
        )
