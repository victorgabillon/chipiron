"""
This module defines the TreeNode class, which represents a node in a tree structure for a chess game.
"""

from dataclasses import dataclass, field
from typing import Any

import chess

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board.iboard import LegalMoveKeyGeneratorP
from chipiron.environments.chess.move.imove import moveKey

from .itree_node import ITreeNode

# todo replace the any with a defaut value in ITReenode when availble in python; 3.13?


@dataclass(slots=True)
class TreeNode[ChildrenType: ITreeNode[Any]]:
    r"""
    The TreeNode class stores information about a specific board position, including the board representation,
    the player to move, the half-move count, and the parent-child relationships with other nodes.

    Attributes:
        id\_ (int): The number to identify this node for easier debugging.
        half_move\_ (int): The number of half-moves since the start of the game to reach the board position.
        board\_ (boards.BoardChi): The board representation of the node.
        parent_nodes\_ (set[ITreeNode]): The set of parent nodes to this node.
        all_legal_moves_generated (bool): A boolean indicating whether all moves have been generated.
        non_opened_legal_moves (set[chess.Move]): The set of non-opened legal moves.
        moves_children\_ (dict[chess.Move, ITreeNode | None]): The dictionary mapping moves to child nodes.
        fast_rep (str): The fast representation of the board.
        player_to_move\_ (chess.Color): The color of the player that has to move in the board.

    Methods:
        __post_init__(): Initializes the TreeNode object after it has been created.
        id(): Returns the id of the node.
        player_to_move(): Returns the color of the player to move.
        board(): Returns the board representation.
        half_move(): Returns the number of half-moves.
        moves_children(): Returns the dictionary mapping moves to child nodes.
        parent_nodes(): Returns the set of parent nodes.
        is_root_node(): Checks if the node is a root node.
        legal_moves(): Returns the legal moves of the board.
        add_parent(new_parent_node: ITreeNode): Adds a parent node to the current node.
        is_over(): Checks if the game is over.
        print_moves_children(): Prints the moves-children links of the node.
        test(): Performs a test on the node.
        dot_description(): Returns the dot description of the node.
        test_all_legal_moves_generated(): Tests if all legal moves have been generated.
        get_descendants(): Returns a dictionary of descendants of the node.
    """

    # id is a number to identify this node for easier debug
    id_: int

    # number of half-moves since the start of the game to get to the board position in self.board
    half_move_: int

    # the node represents a board position. we also store the fast representation of the board.
    board_: boards.IBoard

    # the set of parent nodes to this node. Note that a node can have multiple parents!
    parent_nodes_: dict[ITreeNode[ChildrenType], moveKey]

    # all_legal_moves_generated  is a boolean saying whether all moves have been generated.
    # If true the moves are either opened in which case the corresponding opened node is stored in
    # the dictionary self.moves_children, otherwise it is stored in self.non_opened_legal_moves
    all_legal_moves_generated: bool = False
    non_opened_legal_moves: set[moveKey] = field(default_factory=set)

    # dictionary mapping moves to children nodes. Node is set to None if not created
    moves_children_: dict[moveKey, ChildrenType | None] = field(default_factory=dict)

    # the color of the player that has to move in the board
    player_to_move_: chess.Color = field(default_factory=chess.Color)

    def __post_init__(self) -> None:
        """
        Performs post-initialization tasks for the TreeNode class.

        This method is automatically called after the object is initialized.
        It sets the `fast_rep` attribute based on the board's fast representation
        and assigns the current player to `player_to_move_`.

        Parameters:
            None

        Returns:
            None
        """
        if self.board_:
            self.player_to_move_: chess.Color = self.board_.turn

    @property
    def fast_rep(self) -> boards.boardKey:
        return self.board_.fast_representation

    @property
    def id(self) -> int:
        """
        Returns the ID of the tree node.

        Returns:
            int: The ID of the tree node.
        """
        return self.id_

    @property
    def player_to_move(self) -> chess.Color:
        """
        Returns the color of the player who is to make the next move.

        Returns:
            chess.Color: The color of the player who is to make the next move.
        """
        return self.player_to_move_

    @property
    def board(self) -> boards.IBoard:
        """
        Returns the board associated with this tree node.

        Returns:
            boards.BoardChi: The board associated with this tree node.
        """
        return self.board_

    @property
    def half_move(self) -> int:
        """
        Returns the number of half moves made in the game.

        Returns:
            int: The number of half moves made.
        """
        return self.half_move_

    @property
    def moves_children(self) -> dict[moveKey, ChildrenType | None]:
        """
        Returns a bidirectional dictionary containing the children nodes of the current tree node,
        along with the corresponding chess moves that lead to each child node.

        Returns:
            dict[chess.Move, ITreeNode | None]: A bidirectional dictionary mapping chess moves to
            the corresponding child nodes. If a move does not have a corresponding child node, it is
            mapped to None.
        """
        return self.moves_children_

    @property
    def parent_nodes(self) -> dict[ITreeNode[ChildrenType], moveKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated move.

        :return: A dictionary of parent nodes of the current tree node with associated move.
        """
        return self.parent_nodes_

    def is_root_node(self) -> bool:
        """
        Check if the current node is a root node.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        return not self.parent_nodes

    # understand what is best :
    # 1. this function returns a object of type chess.LegalMoveGenerator
    # 2. of type set of moves
    @property
    def legal_moves(self) -> LegalMoveKeyGeneratorP:
        """
        Returns a generator that yields the legal moves for the current board state.

        Returns:
            chess.LegalMoveGenerator: A generator that yields the legal moves.
        """
        return self.board_.legal_moves

    def add_parent(
        self, move: moveKey, new_parent_node: ITreeNode[ChildrenType]
    ) -> None:
        """
        Adds a new parent node to the current node.

        Args:
            move (chess.Move): the move that led to the node from the new_parent_node
            new_parent_node (ITreeNode): The new parent node to be added.

        Raises:
            AssertionError: If the new parent node is already in the parent nodes set.

        Returns:
            None
        """
        # debug
        assert (
            new_parent_node not in self.parent_nodes
        )  # there cannot be two ways to link the same child-parent
        self.parent_nodes[new_parent_node] = move

    def is_over(self) -> bool:
        """
        Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def print_moves_children(self) -> None:
        """
        Prints the moves-children link of the node.

        This method prints the moves-children link of the node, showing the move and the ID of the child node.
        If a child node is None, it will be displayed as 'None'.

        Returns:
            None
        """
        print(
            "here are the ",
            len(self.moves_children_),
            " moves-children link of node",
            self.id,
            ": ",
            end=" ",
        )
        for move, child in self.moves_children_.items():
            if child is None:
                print(move, child, end=" ")
            else:
                print(move, child.id, end=" ")
        print(" ")

    def test(self) -> None:
        """
        This method is used to test the node.
        It calls the `test_all_legal_moves_generated` method to test all legal moves generated.
        """
        # print('testing node', selbestf.id)
        self.test_all_legal_moves_generated()

    def dot_description(self) -> str:
        """
        Returns a string representation of the node in the DOT format.

        The string includes the node's ID, half move, and board FEN.

        Returns:
            A string representation of the node in the DOT format.
        """
        return (
            "id:"
            + str(self.id)
            + " dep: "
            + str(self.half_move)
            + "\nfen:"
            + str(self.board)
        )

    def test_all_legal_moves_generated(self) -> None:
        """
        Test whether all legal moves are generated correctly.

        This method checks if all legal moves are correctly generated by comparing them with the moves stored in the
        `moves_children_` attribute. If `all_legal_moves_generated` is True, it asserts that each move in `board.legal_moves`
        is either present in `moves_children_` or not present in `non_opened_legal_moves`. If `all_legal_moves_generated`
        is False, it checks if there are any moves in `board.legal_moves` that are not present in `moves_children_`.

        Raises:
            AssertionError: If the generated moves do not match the expected moves.
        """
        # print('test_all_legal_moves_generated')
        move: moveKey
        if self.all_legal_moves_generated:
            for move in self.board.legal_moves:
                assert bool(move in self.moves_children_) != bool(
                    move in self.non_opened_legal_moves
                )
        else:
            move_not_in: list[moveKey] = []
            legal_moves: list[moveKey] = self.board.legal_moves.get_all()
            for move in legal_moves:
                if move not in self.moves_children_:
                    move_not_in.append(move)
            if move_not_in == []:
                pass
                # print('test', move_not_in, list(self.board.get_legal_moves()), self.moves_children)
                # print(self.board)
            assert move_not_in != [] or legal_moves == []
