"""
Sequool
"""
from chipiron.players.treevalue.node_selector.notations_and_statics import zipf_picks
from players.treevalue import trees
from players.treevalue import tree_manager as tree_man
from chipiron.players.treevalue.trees.descendants import RangedDescendants
from chipiron.environments import HALF_MOVE
from .index_computation import UpdateAllIndices
from players.treevalue import nodes

from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructions, OpeningInstructor, \
    create_instructions_to_open_all_moves


class SequoolTree2:

    def __init__(self, environment, board_evaluator, color, arg, board):

        #   self.count_to_open_at_depth=[1]
        self.count_visits_at_depth = []
        super().__init__(environment, board_evaluator, color, arg, board)
        self.root_node.index = 0

    def add_to_dic(self, depth, fen, node):  # todo to be changed!! to descendants no?
        super().add_to_dic(depth, fen, node)
        if depth < len(self.all_nodes_2_not_opened):
            self.all_nodes_2_not_opened[depth].add(node)
        #    self.count_to_open_at_depth[node.depth] += 1
        else:  # depth == len(self.all_nodes_2_not_opened)
            assert (depth == len(self.all_nodes_2_not_opened))
            self.all_nodes_2_not_opened.append({node})
            if depth == len(self.count_visits_at_depth):
                self.count_visits_at_depth.append(1)

        #   self.count_to_open_at_depth.append(1)


class Sequool:
    """
    Sequool Node selector
    """
    opening_instructor: OpeningInstructor
    all_nodes_not_opened: RangedDescendants

    # counting the visits for each half_move
    count_visits: dict[HALF_MOVE, int]

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
            if node.index is not None:
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

        self.update_all_indices(all_nodes_not_opened=self.all_nodes_not_opened)

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
