"""
Sequool

This module contains the implementation of the Sequool node selector. The Sequool node selector is responsible for
choosing the best node to open in a move tree based on various selection strategies.

Classes:
- HalfMoveSelector: Protocol defining the interface for a half-move selector.
- StaticNotOpenedSelector: A node selector that considers the number of visits and selects half-moves based on zipf distribution.
- RandomAllSelector: A node selector that selects half-moves randomly.
- Sequool: The main class implementing the Sequool node selector.

Functions:
- consider_nodes_from_all_lesser_half_moves_in_descendants: Consider all nodes in smaller half-moves than the picked half-move using the descendants object.
- consider_nodes_from_all_lesser_half_moves_in_sub_stree: Consider all nodes in smaller half-moves than the picked half-move using tree traversal.
- consider_nodes_only_from_half_moves_in_descendants: Consider only the nodes at the picked depth.
- get_best_node_from_candidates: Get the best node from a list of candidate nodes based on their exploration index data.
"""

import random
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from chipiron.environments import HalfMove
from chipiron.players.move_selector.treevalue import trees
from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import (
    MaxDepthDescendants,
)
from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import (
    zipf_picks,
    zipf_picks_random,
)
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_moves,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_traversal import (
    get_descendants_candidate_not_over,
)
from chipiron.players.move_selector.treevalue.trees.descendants import Descendants

if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.tree_manager as tree_man


class HalfMoveSelector(Protocol):
    """
    Protocol defining the interface for a half-move selector.
    """

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions"
    ) -> None:
        """
        Update the half-move selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

    def select_half_move(
        self, from_node: AlgorithmNode, random_generator: random.Random
    ) -> HalfMove:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """


@dataclass
class StaticNotOpenedSelector:
    """
    A node selector that considers the number of visits and selects half-moves based on zipf distribution.
    """

    all_nodes_not_opened: Descendants

    # counting the visits for each half_move
    count_visits: dict[HalfMove, int] = field(default_factory=dict)

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions"
    ) -> None:
        """
        Update the node selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

        # Update internal info with the latest tree expansions
        expansion: tree_man.TreeExpansion
        for expansion in latest_tree_expansions:
            if expansion.creation_child_node:
                self.all_nodes_not_opened.add_descendant(expansion.child_node)

            # if a new half_move is being created then init the visits to 1
            # (0 would bug as it would automatically be selected with zipf computation)
            half_move: int = expansion.child_node.half_move
            if half_move not in self.count_visits:
                self.count_visits[half_move] = 1

    def select_half_move(
        self, from_node: AlgorithmNode, random_generator: random.Random
    ) -> HalfMove:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """

        filtered_count_visits: dict[int, int | float] = {
            hm: value
            for hm, value in self.count_visits.items()
            if hm in self.all_nodes_not_opened
        }

        # choose a half move based on zipf
        half_move_picked: int = zipf_picks(
            ranks_values=filtered_count_visits,
            random_generator=random_generator,
            shift=True,
            random_pick=False,
        )

        self.count_visits[half_move_picked] += 1

        return half_move_picked


ConsiderNodesFromHalfMoves = Callable[[HalfMove, AlgorithmNode], list[ITreeNode[Any]]]


def consider_nodes_from_all_lesser_half_moves_in_descendants(
    half_move_picked: HalfMove, from_node: AlgorithmNode, descendants: Descendants
) -> list[ITreeNode[Any]]:
    """
    Consider all the nodes that are in smaller half-moves than the picked half-move using the descendants object.

    Args:
        half_move_picked: The picked half-move.
        from_node: The current node.
        descendants: The descendants object.

    Returns:
        A list of nodes to consider.
    """

    nodes_to_consider: list[ITreeNode[Any]] = []
    half_move: int
    # considering all half-move smaller than the half move picked
    for half_move in filter(lambda hm: hm < half_move_picked + 1, descendants):
        nodes_to_consider += list(descendants[half_move].values())

    return nodes_to_consider


def consider_nodes_from_all_lesser_half_moves_in_sub_stree(
    half_move_picked: HalfMove,
    from_node: AlgorithmNode,
) -> list[ITreeNode[Any]]:
    """
    Consider all the nodes that are in smaller half-moves than the picked half-move using tree traversal.

    Args:
        half_move_picked: The picked half-move.
        from_node: The current node.

    Returns:
        A list of nodes to consider.
    """

    nodes_to_consider: list[ITreeNode[Any]] = get_descendants_candidate_not_over(
        from_tree_node=from_node, max_depth=half_move_picked - from_node.half_move
    )
    return nodes_to_consider


def consider_nodes_only_from_half_moves_in_descendants(
    half_move_picked: HalfMove,
    from_node: AlgorithmNode,
    descendants: Descendants,
) -> list[ITreeNode[Any]]:
    """
    Consider only the nodes at the picked depth.

    Args:
        half_move_picked: The picked half-move.
        from_node: The current node.
        descendants: The descendants object.

    Returns:
        A list of nodes to consider.
    """

    return list(descendants[half_move_picked].values())


@dataclass
class RandomAllSelector:
    """
    A node selector that selects half-moves randomly.
    """

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions"
    ) -> None:
        """
        Update the node selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

    def select_half_move(
        self, from_node: AlgorithmNode, random_generator: random.Random
    ) -> HalfMove:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """

        half_move_picked: int
        # choose a half move based on zipf
        assert isinstance(from_node.exploration_index_data, MaxDepthDescendants)
        max_descendants_depth: int = (
            from_node.exploration_index_data.max_depth_descendants
        )
        if max_descendants_depth:
            depth_picked: int = zipf_picks_random(
                ordered_list_elements=list(range(1, max_descendants_depth + 1)),
                random_generator=random_generator,
            )
            half_move_picked = from_node.half_move + depth_picked
        else:
            half_move_picked = from_node.half_move
        return half_move_picked


def get_best_node_from_candidates(
    nodes_to_consider: list[ITreeNode[Any]],
) -> AlgorithmNode:
    """
    Returns the best node from a list of candidate nodes based on their exploration index and half move.

    Args:
        nodes_to_consider (list[ITreeNode]): A list of candidate nodes to consider.

    Returns:
        AlgorithmNode: The best node from the list of candidates.
    """
    best_node: ITreeNode[Any] = nodes_to_consider[0]
    assert isinstance(best_node, AlgorithmNode)
    assert best_node.exploration_index_data is not None
    best_value = (best_node.exploration_index_data.index, best_node.half_move)

    node: ITreeNode[Any]
    for node in nodes_to_consider:
        assert isinstance(node, AlgorithmNode)
        assert node.exploration_index_data is not None
        if node.exploration_index_data.index is not None:
            assert best_node.exploration_index_data is not None
            if (
                best_node.exploration_index_data.index is None
                or (node.exploration_index_data.index, node.half_move) < best_value
            ):
                best_node = node
                best_value = (node.exploration_index_data.index, node.half_move)
    return best_node


@dataclass
class Sequool:
    """
    The main class implementing the Sequool node selector.
    """

    opening_instructor: OpeningInstructor
    all_nodes_not_opened: Descendants
    recursif: bool
    random_depth_pick: bool
    half_move_selector: HalfMoveSelector
    random_generator: random.Random
    consider_nodes_from_half_moves: ConsiderNodesFromHalfMoves

    def choose_node_and_move_to_open(
        self,
        tree: trees.MoveAndValueTree,
        latest_tree_expansions: "tree_man.TreeExpansions",
    ) -> OpeningInstructions:
        """
        Choose the best node to open in the move tree and return the opening instructions.

        Args:
            tree: The move tree.
            latest_tree_expansions: The latest tree expansions.

        Returns:
            The opening instructions.
        """

        self.half_move_selector.update_from_expansions(
            latest_tree_expansions=latest_tree_expansions
        )

        opening_instructions: OpeningInstructions = (
            self.choose_node_and_move_to_open_recur(from_node=tree.root_node)
        )

        return opening_instructions

    def choose_node_and_move_to_open_recur(
        self, from_node: AlgorithmNode
    ) -> OpeningInstructions:
        """
        Recursively choose the best node to open in the move tree and return the opening instructions.

        Args:
            from_node: The current node.

        Returns:
            The opening instructions.
        """

        half_move_selected: HalfMove = self.half_move_selector.select_half_move(
            from_node=from_node, random_generator=self.random_generator
        )

        nodes_to_consider: list[ITreeNode[Any]] = self.consider_nodes_from_half_moves(
            half_move_selected, from_node
        )

        best_node: AlgorithmNode = get_best_node_from_candidates(
            nodes_to_consider=nodes_to_consider
        )

        if not self.recursif:
            self.all_nodes_not_opened.remove_descendant(best_node)

        if self.recursif and best_node.tree_node.all_legal_moves_generated:
            return self.choose_node_and_move_to_open_recur(from_node=best_node)
        else:
            all_moves_to_open = self.opening_instructor.all_moves_to_open(
                node_to_open=best_node.tree_node
            )
            opening_instructions: OpeningInstructions = (
                create_instructions_to_open_all_moves(
                    moves_to_play=all_moves_to_open, node_to_open=best_node
                )
            )

            return opening_instructions
