"""
Sequool
"""
import random
import typing
from dataclasses import dataclass, field
from typing import Callable
from typing import Protocol

from chipiron.environments import HalfMove
from chipiron.players.move_selector.treevalue import trees
from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import MaxDepthDescendants
from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import zipf_picks, zipf_picks_random
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructions, \
    OpeningInstructor, \
    create_instructions_to_open_all_moves
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.nodes.tree_traversal import get_descendants_candidate_not_over
from chipiron.players.move_selector.treevalue.trees.descendants import Descendants

if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.tree_manager as tree_man


# NodeCandidatesSelector = Callable[[random.Random], list[AlgorithmNode]]


class HalfMoveSelector(Protocol):
    def update_from_expansions(
            self,
            latest_tree_expansions: 'tree_man.TreeExpansions'
    ) -> None:
        ...

    def select_half_move(
            self,
            from_node: AlgorithmNode,
            random_generator
    ) -> HalfMove:
        ...


@dataclass
class StaticNotOpenedSelector:
    all_nodes_not_opened: Descendants

    # counting the visits for each half_move
    count_visits: dict[HalfMove, int] = field(default_factory=dict)

    def update_from_expansions(
            self,
            latest_tree_expansions: 'tree_man.TreeExpansions'
    ) -> None:

        # print('updating the all_nodes_not_opened')
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
            self,
            from_node: AlgorithmNode,
            random_generator
    ) -> HalfMove:

        filtered_count_visits: dict[int, int | float] = {hm: value for hm, value in self.count_visits.items() if
                                                         hm in self.all_nodes_not_opened}

        # choose a half move based on zipf
        half_move_picked: int = zipf_picks(
            ranks_values=filtered_count_visits,
            random_generator=random_generator,
            shift=True,
            random_pick=False
        )

        self.count_visits[half_move_picked] += 1

        return half_move_picked


ConsiderNodesFromHalfMoves = Callable[[HalfMove, AlgorithmNode], list[AlgorithmNode]]


def consider_nodes_from_all_lesser_half_moves_in_descendants(
        half_move_picked: HalfMove,
        from_node: AlgorithmNode,
        descendants: Descendants
) -> list[AlgorithmNode]:
    """ consider all the nodes that are in smaller half moves than the picked half-move
    Done with the descendants objet"""

    nodes_to_consider: list[AlgorithmNode] = []
    half_move: int
    # considering all half-move smaller than the half move picked
    for half_move in filter(lambda hm: hm < half_move_picked + 1, descendants):
        nodes_to_consider += list(descendants[half_move].values())

    return nodes_to_consider


def consider_nodes_from_all_lesser_half_moves_in_sub_stree(
        half_move_picked: HalfMove,
        from_node: AlgorithmNode,
) -> list[AlgorithmNode]:
    """ consider all the nodes that are in smaller half moves than the picked half-move
    Done with tree traversal from the node"""

    nodes_to_consider: list[AlgorithmNode] = get_descendants_candidate_not_over(
        from_tree_node=from_node,
        max_depth=half_move_picked - from_node.half_move
    )
    return nodes_to_consider


def consider_nodes_only_from_half_moves_in_descendants(
        half_move_picked: HalfMove,
        from_node: AlgorithmNode,
        descendants: Descendants,
) -> list[AlgorithmNode]:
    """ consider only the nodes at the picked depth"""
    return list(descendants[half_move_picked].values())


@dataclass
class RandomAllSelector:

    def update_from_expansions(
            self,
            latest_tree_expansions: 'tree_man.TreeExpansions'
    ) -> None:
        ...

    def select_half_move(
            self,
            from_node: AlgorithmNode,
            random_generator
    ) -> HalfMove:
        half_move_picked: int
        # choose a half move based on zipf
        assert isinstance(from_node.exploration_index_data, MaxDepthDescendants)
        max_descendants_depth: int = from_node.exploration_index_data.max_depth_descendants
        if max_descendants_depth:
            depth_picked: int = zipf_picks_random(
                ordered_list_elements=list(range(1, max_descendants_depth + 1)),
                random_generator=random_generator
            )
            half_move_picked = from_node.half_move + depth_picked
        else:
            half_move_picked = from_node.half_move
        return half_move_picked


def get_best_node_from_candidates(
        nodes_to_consider: list[AlgorithmNode]
) -> AlgorithmNode:
    best_node: AlgorithmNode = nodes_to_consider[0]
    assert best_node.exploration_index_data is not None
    best_value = (best_node.exploration_index_data.index, best_node.half_move)

    node: AlgorithmNode
    for node in nodes_to_consider:
        assert node.exploration_index_data is not None
        if node.exploration_index_data.index is not None:
            if best_node.exploration_index_data.index is None \
                    or (node.exploration_index_data.index, node.half_move) < best_value:
                best_node = node
                best_value = (node.exploration_index_data.index, node.half_move)
    return best_node


@dataclass
class Sequool:
    """
    Sequool Node selector
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
            latest_tree_expansions: 'tree_man.TreeExpansions'
    ) -> OpeningInstructions:

        self.half_move_selector.update_from_expansions(
            latest_tree_expansions=latest_tree_expansions
        )

        opening_instructions: OpeningInstructions = self.choose_node_and_move_to_open_recur(
            from_node=tree.root_node
        )
        # print('opening_instruction', opening_instructions)
        return opening_instructions

    def choose_node_and_move_to_open_recur(
            self,
            from_node: AlgorithmNode
    ) -> OpeningInstructions:

        half_move_selected: HalfMove = self.half_move_selector.select_half_move(
            from_node=from_node,
            random_generator=self.random_generator
        )

        nodes_to_consider: list[AlgorithmNode] = self.consider_nodes_from_half_moves(
            half_move_selected,
            from_node
        )

        best_node: AlgorithmNode = get_best_node_from_candidates(nodes_to_consider=nodes_to_consider)

        if not self.recursif:
            self.all_nodes_not_opened.remove_descendant(best_node)

        # print('grn', best_node.tree_node.all_legal_moves_generated, best_node.tree_node.moves_children)
        if self.recursif and best_node.tree_node.all_legal_moves_generated:
            print('RECUUUUR', best_node.half_move)
            return self.choose_node_and_move_to_open_recur(from_node=best_node)
        else:
            all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=best_node.tree_node)
            opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(
                moves_to_play=all_moves_to_open,
                node_to_open=best_node)

            return opening_instructions
