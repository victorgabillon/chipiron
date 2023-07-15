"""
Sequool
"""
from chipiron.players.treevalue.node_selector.notations_and_statics import zipf_picks
import math
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructions, OpeningInstructor
from .. import trees
from .. import tree_manager as tree_man
from chipiron.players.treevalue.nodes.utils import are_all_moves_and_children_opened
from chipiron.players.treevalue.trees.descendants import RangedDescendants


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


def update_all_indices(tree: trees.MoveAndValueTree) -> None:
    for depth in range(tree.get_max_depth()):
        for node in tree.all_nodes[depth].values():
            for child in node.moves_children.values():
                child.index = None  # rootnode is not set to zero haha

    root_node_value_white = tree.root_node.value_white
    root_node_second_value_white = tree.root_node.second_best_child().value_white

    for depth in range(tree.get_max_depth()):
        # print('depth',depth)
        for node in tree.all_nodes[depth].values():
            #   print('node',node.id)
            for child in node.moves_children.values():
                #      print('child', child.id)
                if node.index is None:
                    index = None
                else:
                    if depth % 2 == 0:
                        if self.root_node.best_child() in child.first_moves:  # todo what if it is inboth at the sam time
                            if child == node.best_child():
                                index = abs(child.value_white - root_node_second_value_white) / 2
                            else:
                                index = None
                        else:
                            index = abs(child.value_white - root_node_value_white) / 2
                    else:  # depth %2 ==1
                        if self.root_node.best_child() in child.first_moves:
                            index = abs(child.value_white - root_node_second_value_white) / 2
                        else:  # not the best line
                            if child == node.best_child():
                                index = abs(child.value_white - root_node_value_white) / 2
                            else:  # not the best child response
                                index = None
                if index is not None:
                    if child.index is None:  # if the index has beene initiated already by another parent node
                        child.index = index
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].index is not None)

                    else:
                        child.index = min(child.index, index)
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].index is not None)

    assert (tree.root_node.best_node_sequence[-1].index is not None)


class Sequool:
    """
    Sequool Node selector
    """
    opening_instructor: OpeningInstructor
    all_nodes_not_opened: RangedDescendants

    def __init__(
            self,
            opening_instructor: OpeningInstructor,
            all_nodes_not_opened: RangedDescendants
    ):
        self.opening_instructor = opening_instructor
        self.all_nodes_not_opened = all_nodes_not_opened

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

        expansion: tree_man.TreeExpansion
        for expansion in latest_tree_expansions:
            self.all_nodes_not_opened.add_descendant(expansion.child_node)
            depth: int = tree.node_depth(expansion.child_node)
            if depth == len(self.all_nodes_not_opened) and depth == len(self.count_visits_at_depth):
                self.count_visits_at_depth.append(1)

        if are_all_moves_and_children_opened(tree_node=tree.root_node):
            self.update_all_indices(tree)

        # todo remove the depths that are fully explored
        #   self.tree.save_raw_data_to_file(self.count)
        #  self.count += 1
        #      print('###', self.count)

        depth_picked, best_value = zipf_picks(list_elements=self.all_nodes_not_opened,
                                              value_of_element=lambda depth_: tree.count_visits_at_depth[
                                                  depth_] if self.is_there_smth_to_open(depth_) else math.inf)

        self.tree.count_visits_at_depth[depth_picked] += 1

        nodes_to_consider = list(self.tree.all_nodes_2_not_opened[depth_picked])

        nodes_to_consider = []
        for depth in range(depth_picked + 1):
            nodes_to_consider += list(self.tree.all_nodes_2_not_opened[depth])

        best_node = nodes_to_consider[0]
        best_value = (best_node.index, best_node.half_move)
        for node in nodes_to_consider:
            # print('~',node.index , best_value,node.id,node.depth)

            if node.index is not None:
                if best_node.index is None or (node.index, node.half_move) < best_value:
                    best_node = node
                    best_value = (node.index, node.half_move)

        self.tree.all_nodes_2_not_opened[best_node.half_move].remove(best_node)
        #   self.tree.count_to_open_at_depth=[node.depth] -=1
        assert (best_node.index is not None)
        return self.opening_instructor.instructions_to_open_all_moves(best_node)

    def get_move_from_player(self, board, timetoMove):
        print(board.chess_board)
        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('Sequool2')
