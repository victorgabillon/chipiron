"""
Sequool
"""
from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import zipf_picks
from chipiron.players.move_selector.treevalue import trees
from chipiron.players.move_selector.treevalue import tree_manager as tree_man
from chipiron.players.move_selector.treevalue.trees.descendants import RangedDescendants
from chipiron.environments import HalfMove
from .index_computation import UpdateAllIndices
from ... import nodes

from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructions, \
    OpeningInstructor, \
    create_instructions_to_open_all_moves


class Sequool:
    """
    Sequool Node selector
    """
    opening_instructor: OpeningInstructor
    all_nodes_not_opened: RangedDescendants

    # counting the visits for each half_move
    count_visits: dict[HalfMove, int]

    # the function that updates the indices
    update_all_indices: UpdateAllIndices

    def __init__(
            self,
            opening_instructor: OpeningInstructor,
            all_nodes_not_opened: RangedDescendants,
            update_all_indices: UpdateAllIndices
    ):
        self.opening_instructor = opening_instructor
        self.all_nodes_not_opened = all_nodes_not_opened
        self.count_visits = {}
        self.update_all_indices = update_all_indices

    def is_there_smth_to_open(self, depth):
        res = False
        for node in self.all_nodes_not_opened[depth]:
            if node.exploration_manager.index is not None:
                res = True
                break
        return res

    def choose_node_and_move_to_open(
            self,
            tree: trees.MoveAndValueTree,
            latest_tree_expansions: tree_man.TreeExpansions
    ) -> OpeningInstructions:

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

        self.update_all_indices(tree=tree)



        filtered_count_visits = filter(
            lambda hm: bool(hm in self.all_nodes_not_opened),
            self.count_visits
        )

        # choose a half move based on zipf
        half_move_picked: int = zipf_picks(
            ranks=self.all_nodes_not_opened,
            values=filtered_count_visits,
            shift=True
        )

        self.count_visits[half_move_picked] += 1

        nodes_to_consider: list[nodes.AlgorithmNode] = []
        half_move: int
        # considering all half-move smaller than the half move picked
        for half_move in filter(lambda hm: hm < half_move_picked + 1, self.all_nodes_not_opened):
            nodes_to_consider += list(self.all_nodes_not_opened[half_move].values())

        best_node: nodes.AlgorithmNode = nodes_to_consider[0]
        best_value = (best_node.exploration_manager.index, best_node.half_move)

        node: nodes.AlgorithmNode
        for node in nodes_to_consider:
            if node.exploration_manager.index is not None:

                if best_node.exploration_manager.index is None \
                        or (node.exploration_manager.index, node.half_move) < best_value:
                    best_node = node
                    best_value = (node.exploration_manager.index, node.half_move)

        self.all_nodes_not_opened.remove_descendant(best_node)

        all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=best_node.tree_node)
        opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(
            moves_to_play=all_moves_to_open,
            node_to_open=best_node)

        return opening_instructions
