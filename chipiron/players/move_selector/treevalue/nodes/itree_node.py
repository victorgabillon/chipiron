"""
This module defines the interface for a tree node in a chess move selector.

The `ITreeNode` protocol represents a node in a tree structure used for selecting chess moves.
It provides properties and methods for accessing information about the node, such as its ID,
the chess board state, the half move count, the child nodes, and the parent nodes.

The `ITreeNode` protocol also defines methods for adding a parent node, generating a dot description
for visualization, checking if all legal moves have been generated, accessing the legal moves,
and checking if the game is over.

Note: This is an interface and should not be instantiated directly.
"""

from __future__ import annotations

from typing import Any, Protocol

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board.iboard import LegalMoveKeyGeneratorP
from chipiron.environments.chess.move.imove import moveKey

# to force Any to stay because of weird interaction between flake8 and pycharm
a: Any = 0


class ITreeNode[T: ITreeNode[Any]](Protocol):
    """
    The `ITreeNode` protocol represents a node in a tree structure used for selecting chess moves.
    """

    @property
    def id(self) -> int:
        """
        Get the ID of the node.

        Returns:
            The ID of the node.
        """

    # actually giving access to the boars gives access to a lot of sub fucntion so might
    # be no need to ask for them in the interfacec expicitly
    @property
    def board(self) -> boards.IBoard:
        """
        Get the chess board state of the node.

        Returns:
            The chess board state of the node.
        """

    @property
    def half_move(self) -> int:
        """
        Get the half move count of the node.

        Returns:
            The half move count of the node.
        """

    @property
    def moves_children(self) -> dict[moveKey, T | None]:
        """
        Get the child nodes of the node.

        Returns:
            A bidirectional dictionary mapping chess moves to child nodes.
        """

    @property
    def parent_nodes(self) -> dict[ITreeNode[T], moveKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated move.

        :return: A dictionary of parent nodes of the current tree node with associated move.
        """

    def add_parent(self, move: moveKey, new_parent_node: ITreeNode[T]) -> None:
        """
        Add a parent node to the node.

        Args:
            new_parent_node: The parent node to add.
            move (chess.Move): the move that led to the node from the new_parent_node

        """

    def dot_description(self) -> str:
        """
        Generate a dot description for visualization.

        Returns:
            A string containing the dot description.
        """

    @property
    def all_legal_moves_generated(self) -> bool:
        """
        Check if all legal moves have been generated.

        Returns:
            True if all legal moves have been generated, False otherwise.
        """

    @all_legal_moves_generated.setter
    def all_legal_moves_generated(self, value: bool) -> None:
        """
        Set the flag indicating that all legal moves have been generated.
        """

    @property
    def legal_moves(self) -> LegalMoveKeyGeneratorP:
        """
        Get the legal moves of the node.

        Returns:
            A generator for iterating over the legal moves.
        """

    @property
    def fast_rep(self) -> boards.boardKey:
        """
        Get the fast representation of the node.

        Returns:
            The fast representation of the node as a string.
        """

    def is_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            True if the game is over, False otherwise.
        """
