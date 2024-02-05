"""
Sequool
"""
from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import zipf_picks, zipf_picks_random
from chipiron.players.move_selector.treevalue import trees
from chipiron.players.move_selector.treevalue import tree_manager as tree_man
from chipiron.players.move_selector.treevalue.trees.descendants import Descendants, RangedDescendants
from chipiron.environments import HalfMove

from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructions, \
    OpeningInstructor, \
    create_instructions_to_open_all_moves

from dataclasses import dataclass, field
import random
import chipiron.players.move_selector.treevalue.nodes as nodes
from typing import Protocol


# NodeCandidatesSelector = Callable[[random.Random], list[nodes.AlgorithmNode]]


class ExplorationNodeSelector(Protocol):
    def update_from_expansions(
            self,
            latest_tree_expansions: tree_man.TreeExpansions
    ) -> None:
        ...

    def select(
            self,
            random_generator
    ) -> list[nodes.AlgorithmNode]:
        ...


@dataclass
class StaticNotOpenedSelector:
    all_nodes_not_opened: Descendants

    # counting the visits for each half_move
    count_visits: dict[HalfMove, int] = field(default_factory=dict)

    def update_from_expansions(
            self,
            latest_tree_expansions: tree_man.TreeExpansions
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

    def select(
            self,
            random_generator
    ) -> list[nodes.AlgorithmNode]:

        filtered_count_visits = {hm: value for hm, value in self.count_visits.items() if hm in self.all_nodes_not_opened}

        #print('tt', self.count_visits,
       #       filtered_count_visits)

        # choose a half move based on zipf
        half_move_picked: int = zipf_picks(
            ranks_values=filtered_count_visits,
            random_generator=random_generator,
            shift=True,
            random_pick=False
        )
        #print('half_move_picked', half_move_picked, list(filtered_count_visits), self.count_visits,
       #       self.all_nodes_not_opened)
        import time
        #time.sleep(2)
        self.count_visits[half_move_picked] += 1

        nodes_to_consider: list[nodes.AlgorithmNode] = []
        half_move: int
        # considering all half-move smaller than the half move picked
        for half_move in filter(lambda hm: hm < half_move_picked + 1, self.all_nodes_not_opened):
            nodes_to_consider += list(self.all_nodes_not_opened[half_move].values())

        return nodes_to_consider


@dataclass
class RandomAllSelector:
    all_nodes: RangedDescendants

    # counting the visits for each half_move
    count_visits: dict[HalfMove, int] = field(default_factory=dict)

    def update_from_expansions(
            self,
            latest_tree_expansions: tree_man.TreeExpansions
    ) -> None:
        ...

    def select(
            self,
            random_generator
    ) -> list[nodes.AlgorithmNode]:
        # choose a half move based on zipf
        half_move_picked: int = zipf_picks(
            ranks=self.all_nodes_not_opened,
            values=filtered_count_visits,
            random_generator=random_generator,
            shift=True,
            random_pick=True
        )

        self.count_visits[half_move_picked] += 1

        nodes_to_consider: list[nodes.AlgorithmNode] = []
        half_move: int
        # considering all half-move smaller than the half move picked
        for half_move in filter(lambda hm: hm < half_move_picked + 1, self.all_nodes_not_opened):
            nodes_to_consider += list(self.all_nodes_not_opened[half_move].values())

        return nodes_to_consider


@dataclass
class Sequool:
    """
    Sequool Node selector
    """
    opening_instructor: OpeningInstructor
    all_nodes_not_opened: RangedDescendants
    recursif: bool
    random_depth_pick: bool
    node_candidates_selector: ExplorationNodeSelector
    random_generator: random.Random

    def choose_node_and_move_to_open(
            self,
            tree: trees.MoveAndValueTree,
            latest_tree_expansions: tree_man.TreeExpansions
    ) -> OpeningInstructions:

        # print('rrrtttr',latest_tree_expansions)
        self.node_candidates_selector.update_from_expansions(
            latest_tree_expansions=latest_tree_expansions
        )

        nodes_to_consider: list[nodes.AlgorithmNode] = self.node_candidates_selector.select(
            random_generator=self.random_generator
        )

        best_node: nodes.AlgorithmNode = nodes_to_consider[0]
        best_value = (best_node.exploration_index_data.index, best_node.half_move)

        node: nodes.AlgorithmNode
        for node in nodes_to_consider:
            if node.exploration_index_data.index is not None:
               # print('index', node.id, node.exploration_index_data.index)

                if best_node.exploration_index_data.index is None \
                        or (node.exploration_index_data.index, node.half_move) < best_value:
                    best_node = node
                    best_value = (node.exploration_index_data.index, node.half_move)

       # print('defrs', best_node.id, self.recursif)

        import time
        # time.sleep(2)
        if not self.recursif:
            self.all_nodes_not_opened.remove_descendant(best_node)

        all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=best_node.tree_node)
        opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(
            moves_to_play=all_moves_to_open,
            node_to_open=best_node)

        if self.recursif and best_node.tree_node.all_legal_moves_generated:
            return self.choose_node_and_move_to_open(tree=tree,
                                                     latest_tree_expansions=[])
        else:
            return opening_instructions
