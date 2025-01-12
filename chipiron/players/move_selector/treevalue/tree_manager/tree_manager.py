"""
This module contains the TreeManager class, which is responsible for managing a tree by opening new nodes and updating the values and indexes on the nodes.
"""

import typing
from typing import Any

import chipiron.environments.chess.board as board_mod
import chipiron.players.move_selector.treevalue.nodes as node
import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.node_factory.node_factory import (
    TreeNodeFactory,
)
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import (
    OpeningInstructions,
)
from chipiron.players.move_selector.treevalue.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)

# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999
if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.node_selector as node_sel


class TreeManager:
    """
    This class manages a tree by opening new nodes and updating the values and indexes on the nodes.
    """

    node_factory: TreeNodeFactory[node.ITreeNode[Any]]

    def __init__(self, node_factory: TreeNodeFactory[node.ITreeNode[Any]]) -> None:
        self.node_factory = node_factory

    def open_node_move(
        self,
        tree: trees.MoveAndValueTree,
        parent_node: node.ITreeNode[Any],
        move: moveKey,
    ) -> TreeExpansion:
        """
        Opening a Node that contains a board following a move.

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            move: The move to play to expand the node.

        Returns:
            The tree expansion object.
        """
        # The parent board is copied, we only copy the stack (history of previous board) if the depth is smaller than 2
        # Having the stack information allows checking for draw by repetition.
        # To limit computation we limit copying it all the time. The resulting policy will only be aware of immediate
        # risk of draw by repetition
        copy_stack: bool = tree.node_depth(parent_node) < 2
        board: board_mod.IBoard = parent_node.board.copy(
            stack=copy_stack,
            deep_copy_legal_moves=False,  # trick to win time (the original legal moves is assume to not be changed as
            # moves are not supposed to be played anymore on that board and therefore this allows copy by reference
        )

        # The move is played. The board is now a new board
        modifications: board_mod.BoardModificationP | None = board.play_move_key(
            move=move
        )

        return self.open_node(
            tree=tree,
            parent_node=parent_node,
            board=board,
            modifications=modifications,
            move=move,
        )

    def open_node(
        self,
        tree: trees.MoveAndValueTree,
        parent_node: node.ITreeNode[Any],
        board: board_mod.IBoard,
        modifications: board_mod.BoardModificationP | None,
        move: moveKey,
    ) -> TreeExpansion:
        """
        Opening a Node that contains a board given the modifications.
        Checks if the new node needs to be created or if the new_board already existed in the tree
         (was reached from a different serie of move)

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            board: The board object that is a move forward compared to the board in the parent node
            modifications: The board modifications.
            move: The move to play to expand the node.

        Returns:
            The tree expansion object.
        """

        # Creation of the child node. If the board already exited in another node, that node is returned as child_node.
        half_move: int = parent_node.half_move + 1
        fast_rep: board_mod.boardKey = board.fast_representation

        child_node: node.ITreeNode[Any]
        need_creation_child_node: bool = (
            tree.descendants.is_new_generation(half_move)
            or fast_rep not in tree.descendants.descendants_at_half_move[half_move]
        )
        if need_creation_child_node:
            child_node = self.node_factory.create(
                board=board,
                half_move=half_move,
                count=tree.nodes_count,
                move_from_parent=move,
                parent_node=parent_node,
                modifications=modifications,
            )
            tree.nodes_count += 1
            tree.descendants.add_descendant(
                child_node
            )  # add it to the list of descendants

        else:  # the node already exists
            child_node = tree.descendants[half_move][fast_rep]
            child_node.add_parent(move=move, new_parent_node=parent_node)

        tree_expansion: TreeExpansion = TreeExpansion(
            child_node=child_node,
            parent_node=parent_node,
            board_modifications=modifications,
            creation_child_node=need_creation_child_node,
            move=move,
        )

        # add it to the list of opened move and out of the non-opened moves
        parent_node.moves_children[move] = tree_expansion.child_node
        #   parent_node.tree_node.non_opened_legal_moves.remove(move)
        tree.move_count += 1  # counting moves

        return tree_expansion

    def open_instructions(
        self, tree: trees.MoveAndValueTree, opening_instructions: OpeningInstructions
    ) -> TreeExpansions:
        """
        Opening multiple nodes based on the opening instructions.

        Args:
            tree: The tree object.
            opening_instructions: The opening instructions.

        Returns:
            The tree expansions that have been performed.
        """

        # place to store the tree expansion logs generated by the openings
        tree_expansions: TreeExpansions = TreeExpansions()

        opening_instruction: node_sel.OpeningInstruction
        for opening_instruction in opening_instructions.values():
            # open
            tree_expansion: TreeExpansion = self.open_node_move(
                tree=tree,
                parent_node=opening_instruction.node_to_open,
                move=opening_instruction.move_to_play,
            )

            # concatenate the tree expansions
            tree_expansions.add(tree_expansion=tree_expansion)

        return tree_expansions

    def print_some_stats(
        self,
        tree: trees.MoveAndValueTree,
    ) -> None:
        """
        Print some statistics about the tree.

        Args:
            tree: The tree object.
        """
        print(
            "Tree stats: move_count",
            tree.move_count,
            " node_count",
            tree.descendants.get_count(),
        )
        sum_ = 0
        tree.descendants.print_stats()
        for half_move in tree.descendants:
            sum_ += len(tree.descendants[half_move])
            print("half_move", half_move, len(tree.descendants[half_move]), sum_)

    def test_count(
        self,
        tree: trees.MoveAndValueTree,
    ) -> None:
        """
        Test the count of nodes in the tree.

        Args:
            tree: The tree object.
        """
        assert tree.descendants.get_count() == tree.nodes_count

    def print_best_line(
        self,
        tree: trees.MoveAndValueTree,
    ) -> None:
        """
        Print the best line in the tree.

        Args:
            tree: The tree object.
        """
        raise Exception("should not be called no? Think about modifying...")
