"""
This module provides functions and classes for creating board representations used in neural networks.

The main classes in this module are:
- `Representation364`: Represents the board state using a tensor of size 364.
- `Representation364Factory`: Factory class for creating `Representation364` objects.

The module also includes helper functions for converting board states to tensors and vice versa.

Note: This module is part of the `chipiron` package.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board.iboard import IBoard
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode

from .board_representation import BoardRepresentation


class CreateFromBoard[T_BoardRepresentation: BoardRepresentation](Protocol):
    def __call__(self, board: boards.IBoard) -> T_BoardRepresentation: ...


class CreateFromBoardAndFromParent[T_BoardRepresentation: BoardRepresentation](
    Protocol
):
    def __call__(
        self,
        board: IBoard,
        board_modifications: boards.BoardModificationP,
        parent_node_board_representation: T_BoardRepresentation,
    ) -> T_BoardRepresentation: ...


@dataclass
class RepresentationFactory[T_BoardRepresentation: BoardRepresentation]:
    """
    Factory class  for creating instances of BoardFactory
    """

    create_from_board: CreateFromBoard[T_BoardRepresentation]
    create_from_board_and_from_parent: CreateFromBoardAndFromParent[
        T_BoardRepresentation
    ]

    def create_from_transition(
        self,
        tree_node: TreeNode[Any],
        parent_node_representation: T_BoardRepresentation | None,
        modifications: boards.BoardModificationP | None,
    ) -> T_BoardRepresentation:
        """
        Create a Representation364 object from a transition.

        Args:
            tree_node (TreeNode): The current tree node.
            parent_node (AlgorithmNode | None): The parent node of the current tree node.
            modifications (board_mod.BoardModification | None): The board modifications.

        Returns:
            Representation364: The created Representation364 object.
        """
        """  this version is supposed to be faster as it only modifies the parent
        representation with the last move and does not scan fully the new board"""
        if parent_node_representation is None:  # this is the root_node
            representation = self.create_from_board(board=tree_node.board)
        else:
            if modifications is None:
                representation = self.create_from_board(board=tree_node.board)
            else:
                representation = self.create_from_board_and_from_parent(
                    board=tree_node.board,
                    board_modifications=modifications,
                    parent_node_board_representation=parent_node_representation,
                )

        return representation
