from Players.TreeAndValueBuilders.TreeAndValue import TreeAndValue
from Players.TreeAndValueBuilders.Trees.MoveAndValueTree import MoveAndValueTree
from Players.TreeAndValueBuilders.Trees.Nodes.index_tree_node import IndexTreeNode
from Players.TreeAndValueBuilders.notations_and_statics import zipf_picks
from Players.TreeAndValueBuilders.Trees.descendants import Descendants
from Players.TreeAndValueBuilders.Trees.Nodes.proportion_tree_node import VisitsAndProportionsNode
from Players.TreeAndValueBuilders.Trees.Nodes.tree_node_with_descendants import NodeWithDescendants
import math


class ZipfSequoolTreeNode2(VisitsAndProportionsNode, IndexTreeNode):
    pass


class IndexDescendantsTreeNode(NodeWithDescendants, IndexTreeNode):
    pass


class ZipfSequoolTree2(MoveAndValueTree):

    def __init__(self, environment, board_evaluator, color, arg, board):

        self.descendants_not_opened = Descendants()
        self.descendants_not_opened_not_none = Descendants()
        self.count_visits_at_depth = []
        super().__init__(environment, board_evaluator, color, arg, board)
        self.root_node.index = 0

    def create_tree_node(self, board, half_move, count, father_node):
        if half_move == self.half_move:
            return ZipfSequoolTreeNode2(board, half_move, count, father_node, self.zipf_style)
        if half_move == self.half_move + 1:
            return IndexDescendantsTreeNode(board, half_move, count, father_node)
        else:
            return IndexTreeNode(board, half_move, count, father_node)

    def update_after_link_creation(self, node, parent_node):
        for first in parent_node.first_moves:
            if first not in node.first_moves:
                first.descendants.add_descendant(node)

    def update_after_node_creation(self, node):
        self.descendants_not_opened.add_descendant(node)

        for node_ in node.first_moves:
            if node_.best_node_sequence:
                assert (node_.best_node_sequence[-1] in node_.descendants.descendants_at_half_move[
                    node_.best_node_sequence[-1].half_move].values())
            if node != node_:
                node_.descendants.add_descendant(node)

        if node.half_move > self.root_node.half_move + 1:
            self.root_node.descendants.add_descendant(node)

        root_node_half_move = self.root_node.half_move

        if node.half_move - root_node_half_move - 1 == len(self.count_visits_at_depth):
            self.count_visits_at_depth.append(1)

    def update_all_indices(self):
        print('fff')
        self.descendants_not_opened_not_none = Descendants()

        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    child.index = None  # rootnode is not set to zero haha

        root_node_value_white = self.root_node.value_white
        root_node_second_value_white = self.root_node.second_best_child().value_white

        for half_move in self.descendants:
            # print('depth',depth)
            for node in self.descendants[half_move].values():
                # print('node',node.id)
                for child in node.moves_children.values():
                    # print('child', child.id)
                    if node.index is None:
                        index = None
                    else:
                        if node.player_to_move == self.root_node.player_to_move:
                            if self.root_node.best_child() in child.first_moves:  # todo what if it is inboth at the sam time
                                if child == node.best_child():
                                    index = abs(child.value_white - root_node_second_value_white) / 2
                                else:
                                    index = None
                            else:
                                index = abs(child.value_white - root_node_value_white) / 2
                        else:
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
                            if child.id == self.root_node.best_node_sequence[-1].id:
                                assert (self.root_node.best_node_sequence[-1].index is not None)
                        else:
                            child.index = min(child.index, index)
                            if child.id == self.root_node.best_node_sequence[-1].id:
                                assert (self.root_node.best_node_sequence[-1].index is not None)

                    if child.index is not None:  #
                        if child.half_move in self.descendants_not_opened and child.fast_rep in \
                                self.descendants_not_opened[child.half_move]:
                            if child.half_move not in self.descendants_not_opened_not_none or child.fast_rep not in \
                                    self.descendants_not_opened_not_none.descendants_at_half_move[child.half_move]:
                                self.descendants_not_opened_not_none.add_descendant(child)

        assert (self.root_node.best_node_sequence[-1].index is not None)
        for child in self.root_node.children_not_over:
            if child.best_node_sequence:
                assert (child.best_node_sequence[-1].index is not None)


class ZipfSequool2(TreeAndValue):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return ZipfSequoolTree2(self.environment, self.board_evaluator, self.color, self.arg, board)

    def is_there_smth_to_open(self, half_move):
        res = False
        for node in self.tree.all_nodes_2_not_opened[half_move]:
            if node.index is not None:
                res = True
                break
        return res

    def who_is_there_smth_to_open_old(self, half_move):
        res = set()
        if half_move not in self.tree.descendants_not_opened:
            return set()
        for node in self.tree.descendants_not_opened[half_move].values():
            if node.index is not None:
                res.add(node)
        return res

    def who_is_there_smth_to_open(self, half_move):
        if half_move not in self.tree.descendants_not_opened_not_none:
            return set()
        else:
            return set(self.tree.descendants_not_opened_not_none[half_move].values())

    def choose_node_and_move_to_open(self):

        root_node_half_move = self.tree.root_node.half_move
        self.count_refresh_index += 1
        if self.tree.root_node.are_all_moves_and_children_opened():
            # print('fg')
            if self.count_refresh_index < 10 or self.count_refresh_index % 100 == 0:
                self.tree.update_all_indices()
        node = self.tree.root_node

        if node.moves_children == {}:
            return self.opening_instructor.instructions_to_open_all_moves(node)

        node = node.choose_child_with_visits_and_proportions()

        func = lambda depth_: self.tree.count_visits_at_depth[depth_] \
            if set(node.descendants[1 + root_node_half_move + depth_].values()).intersection(
            self.who_is_there_smth_to_open(1 + root_node_half_move + depth_)) else math.inf

        nodes_to_consider = []
        while not nodes_to_consider:
            depth_picked, best_value = zipf_picks(list_elements=list(node.descendants.keys()),
                                                  value_of_element=func)

            # print([func(i) for i in range(len(list(node.descendants.keys())))])
            # print('depthpicked', depth_picked)
            # self.tree.root_node.descendants.print_stats()

            nodes_to_consider = list(
                set(node.descendants[1 + root_node_half_move + depth_picked].values()).intersection(
                    self.who_is_there_smth_to_open(1 + root_node_half_move + depth_picked)))

            # nodes_to_consider = []
            # for depth in range(depth_picked + 1):
            #     nodes_to_consider += list(
            #         node.get_descendants_at_depth(depth).intersection(self.tree.all_nodes_2_not_opened[depth]))

            if not nodes_to_consider:
                self.tree.update_all_indices()

        best_node = nodes_to_consider[0]
        print('node to consider', len(nodes_to_consider))
        best_value = (best_node.index, best_node.half_move, best_node.id)
        for node in nodes_to_consider:
            if node.index is not None:
                if best_node.index is None or (node.index, node.half_move, node.id) < best_value:
                    best_node = node
                    best_value = (node.index, node.half_move, node.id)

        # print('best_value',best_value)
        print('best_node', best_node.id, best_node.half_move)

        self.tree.descendants_not_opened.remove_descendant(best_node.half_move, best_node.fast_rep, best_node)
        self.tree.descendants_not_opened_not_none.remove_descendant(best_node.half_move, best_node.fast_rep, best_node)

        assert (best_node.index is not None)
        return self.opening_instructor.instructions_to_open_all_moves(best_node)

    def get_move_from_player(self, board, timetoMove):
        print(board.chess_board)
        self.count_refresh_index = 0
        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('ZipfSequool2')
