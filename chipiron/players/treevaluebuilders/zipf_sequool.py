from players.treevaluebuilders.tree_and_value import TreeAndValue
from players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from players.treevaluebuilders.trees.nodes.index_tree_node import IndexTreeNode
from players.treevaluebuilders.notations_and_statics import zipf_picks_random
from players.treevaluebuilders.trees.nodes.proportion_tree_node import VisitsAndProportionsNode
from players.treevaluebuilders.trees.descendants import SortedDescendants
from players.treevaluebuilders.trees.nodes.tree_node_with_descendants import NodeWithDescendantsNoUpdate
from players.treevaluebuilders.trees.first_moves import FirstMoves
import math
import random


class ZipfSequoolTreeNode(VisitsAndProportionsNode, IndexTreeNode):
    pass


class IndexDescendantsTreeNode(NodeWithDescendantsNoUpdate, IndexTreeNode):
    pass


class ZipfSequoolTree(MoveAndValueTree):

    def __init__(self, environment, board_evaluator, arg, board):
        super().__init__(environment, board_evaluator, arg, board)
        self.first_moves = FirstMoves()
        self.count_visits_at_depth = {}

    def create_tree_node(self, board, half_move, count, father_node):
        #print('££££££££dd', count)
        # assert(count !=904)
        if half_move == self.tree_root_half_move:
            new_node = ZipfSequoolTreeNode(board, half_move, count, father_node, self.arg['zipf_style'])
        elif half_move == self.tree_root_half_move + 1:
            new_node = IndexDescendantsTreeNode(board, half_move, count, father_node)
        else:
            new_node = IndexTreeNode(board, half_move, count, father_node)
        node_depth = half_move - self.tree_root_half_move
        if node_depth > 1:
            self.descendants.add_descendant(new_node)
        return new_node

    def update_after_either_node_or_link_creation(self, node, parent_node):

        root_node_half_move = self.tree_root_half_move
        node_depth = node.half_move - self.tree_root_half_move
        if node_depth > 0:
            for first_move in self.first_moves[node]:
                if node.half_move - root_node_half_move - 1 == len(self.count_visits_at_depth[first_move]):
                    self.count_visits_at_depth[first_move].append(1)
        #
        # if node_depth > 1:
        #     first_moves = self.root_node.moves_children.values()
        #     for first_move in first_moves:
        #         if first_move.all_legal_moves_generated:
        #             for hm in first_move.descendants_not_opened.sorted_descendants_at_half_move:
        #                 for descendant in first_move.descendants_not_opened.sorted_descendants_at_half_move[hm]:
        #                     if descendant.all_legal_moves_generated:
        #                         print('%d%', descendant.id)
        #                         assert (not descendant.all_legal_moves_generated)

    def update_after_link_creation(self, node, parent_node):
        previous_first_move = self.first_moves[node].copy()

        self.first_moves.add_first_move(node, parent_node)
        new_first_moves = self.first_moves[node].difference(previous_first_move)
        if new_first_moves:
            node_descendants = node.get_descendants()
        for new_first_move in new_first_moves:
            for descendant in node_descendants:
                if not new_first_move.descendants.contains_node(descendant):
                    new_first_move.descendants.add_descendant(descendant)
            node_descendants_not_opened = node.get_not_opened_descendants()
            for descendant_not_opened in node_descendants_not_opened:
                if not new_first_move.descendants_not_opened.contains_node(descendant_not_opened):
                    index = descendant_not_opened.index if descendant_not_opened.index is not None else math.inf
                    value = (index, descendant_not_opened.half_move, descendant_not_opened.id)
                    new_first_move.descendants_not_opened.add_descendant(descendant_not_opened, value)

    def update_after_node_creation(self, node, parent_node):

        self.first_moves.add_first_move(node, parent_node)
        node_depth = node.half_move - self.tree_root_half_move

        if node_depth == 1:
            if node not in self.count_visits_at_depth:
                self.count_visits_at_depth[node] = []

        node_depth = node.half_move - self.tree_root_half_move

        if node_depth == 1:
            node.descendants_not_opened = SortedDescendants()

        if node_depth >= 1:
            for first_move in self.first_moves[node]:
                first_move.descendants_not_opened.add_descendant(node, (math.inf, node.half_move, node.id))
        #            node.print_a_move_sequence_from_root()
        #            assert(node.id !=30)
        # node.descendants_not_opened.print_info()

        if node_depth >= 2:  # todo can we do this strategy without the descendnats at each first child?
            for first_move in self.first_moves[node]:
                first_move.descendants.add_descendant(node)

        # if self.root_node is not None:
        #     first_moves = self.root_node.moves_children.values()
        #     for f in first_moves:
        #         f.descendants.test_2(f)

    def update_all_indices(self):
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    child.index = None  # rootnode is not set to zero haha

        self.root_node.index = 0
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    index = max(abs(child.get_value_white() - node.get_value_white()) / 2, node.index)
                    index = abs(child.get_value_white() - node.get_value_white()) / 2 +  node.index  #todo why not sqaured?

                    if child.index is None:
                        child.index = index
                    else:
                        child.index = min(child.index, index)
                    for fm in self.first_moves[child]:
                        if not child.all_legal_moves_generated:
                            fm.descendants_not_opened.update_value(child, (child.index, child.half_move, child.id))


        for half_move in self.descendants:
            # print('eee',self.get_max_depth())
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    assert (child.index is not None)
                # for fm in self.first_moves[node]:
                #     if not node.all_legal_moves_generated:
                #         #                            print('hj',fm.id)
                #
                #         #                           print('node', node.id,)
                #         #                          fm.descendants_not_opened.print_info()
                #         #                         print('@',fm.descendants_not_opened.sorted_descendants_at_half_move[half_move])
                #         #                        print('node',node.id,node.index ,fm.descendants_not_opened.sorted_descendants_at_half_move[half_move][node][0])
                #         assert (node.index ==
                #                 fm.descendants_not_opened.sorted_descendants_at_half_move[half_move][node][0])
    #   self.print_some_stats()


class ZipfSequool(TreeAndValue):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return ZipfSequoolTree(self.environment, self.board_evaluators_wrapper, self.arg, board)

    def who_is_there_smth_to_open(self, node, half_move):
        if half_move not in node.descendants_not_opened:
            #  print('~~')
            return {}
        else:
            # print('~s~', type(node.descendants_not_opened),
            #      type(node.descendants_not_opened.sorted_descendants_at_half_move[half_move]))
            return node.descendants_not_opened.sorted_descendants_at_half_move[half_move]

    def candidates_at_depth(self, node, depth_):
        root_node_half_move = self.tree.root_node.half_move
        return self.who_is_there_smth_to_open(node, 1 + root_node_half_move + depth_)  # todo this +1 is a bit confusing

    def how_many_opened_at_depth(self, node, depth_):
        root_node_half_move = self.tree.root_node.half_move
        half_move = 1 + root_node_half_move + depth_
        if half_move not in node.descendants_not_opened:
            return node.descendants.number_of_descendants_at_half_move[half_move]
        else:
            return node.descendants.number_of_descendants_at_half_move[half_move]
            - node.descendants_not_opened.sorted_descendants_at_half_move[half_move]

    def choose_node_and_move_to_open(self):
        # self.tree.update_all_indices()

        # self.tree.root_node.print_proportions()
        self.count_refresh_index += 1
        if self.tree.root_node.are_all_moves_and_children_opened():
            if self.count_refresh_index < 10 or self.count_refresh_index == self.next_update:
                if self.count_refresh_index == self.next_update:
                    self.next_update = int(1.3 * self.next_update + 1)
                print('##000000000000#######', )
                self.tree.update_all_indices()

        node = self.tree.root_node
        if node.moves_children == {}:
            return self.opening_instructor.instructions_to_open_all_moves(node)

        last_node_in_best_line = self.tree.root_node.best_node_sequence[-1]
        if last_node_in_best_line.board.is_attacked(not last_node_in_best_line.player_to_move):
            print('best line is underattacked')
            if random.random()>.5:
                #todo having to do this remove is ugle and needs to be repeated in all retrun please change!!
                for nfm in self.tree.first_moves[last_node_in_best_line]:
                    nfm.descendants_not_opened.remove_descendant(last_node_in_best_line)
                return self.opening_instructor.instructions_to_open_all_moves(last_node_in_best_line)

        #node_first_move = node.choose_child_with_visits_and_proportions()
        node_first_move = node.choose_child_with_proportions()

        print('firstmove', self.tree.root_node.moves_children.inverse[node_first_move])

        def func(depth_):
            # print('depth', depth_, self.tree.count_visits_at_depth[node_first_move])
            candidates = len(self.candidates_at_depth(node_first_move, depth_))
            if candidates:
                return self.how_many_opened_at_depth(node_first_move, depth_)
                # return self.tree.count_visits_at_depth[node_first_move][depth_]
            else:
                return math.inf

        def func_2(depth_):
            # print('depth', depth_, self.tree.count_visits_at_depth[node_first_move])
            candidates = len(self.candidates_at_depth(node_first_move, depth_))
            if candidates:
                return True
                # return self.tree.count_visits_at_depth[node_first_move][depth_]
            else:
                return False

        best_line = self.tree.root_node.best_node_sequence
        best_line_from_selected_first_move = node_first_move.best_node_sequence
        list_elements_ = list(node_first_move.descendants.keys())[0: 1 + len(best_line_from_selected_first_move)]
        # print('%%%',self.tree.count_visits_at_depth[node_first_move],node_first_move.descendants.keys(),
        #       list(node_first_move.descendants.keys())[0: 1 + len(best_line_from_selected_first_move)],
        #       best_line_from_selected_first_move)
        #
        # print('********', list_elements_, node_first_move.id,
        #       self.tree.root_node.moves_children.inverse[node_first_move], best_line_from_selected_first_move,
        #       best_line)

        # print('@@',best_line[-1].moves_children,best_line[-1].are_all_moves_and_children_opened(),best_line[-1].all_legal_moves_generated,best_line[-1].best_node_sequence)
        depth_picked, best_val = zipf_picks_random(list_elements=list_elements_,
                                                   bool_func=func_2)
        print('depth picked', depth_picked, list_elements_)
        # print('no_id', node_first_move.id)
        #  node_first_move.descendants.print_info()
        # node_first_move.descendants_not_opened.print_info()

        # print('values', [func(i) for i in range(len(list(node_first_move.descendants.keys())))])

        #  self.tree.count_visits_at_depth[node_first_move][depth_picked] += 1
        # print('####', type(self.candidates_at_depth(node_first_move, depth_picked)))

        nodes_to_consider_dict = self.candidates_at_depth(node_first_move, depth_picked)
        nodes_to_consider_list = list(nodes_to_consider_dict.keys())  # todo loosing time in the list conversion?

        # print('####', type(nodes_to_consider_list))
        # for i in nodes_to_consider:
        #    print('nodes', i.id)
        # todo check if mixing depth improves results
       # node_first_move.descendants_not_opened.print_info()
       # print(nodes_to_consider_dict)
       # print('best_n4444', nodes_to_consider_dict)

        best_node = nodes_to_consider_list[0]
        # print('op', type(best_node), best_node, type(nodes_to_consider_list[0]))
        index = best_node.index if best_node.index is not None else math.inf
        best_value = (index, best_node.half_move, best_node.id)
        for node_to_consider in nodes_to_consider_list:
            index = node_to_consider.index if node_to_consider.index is not None else math.inf
        #    print('~', node_to_consider.index, (index, node_to_consider.half_move, node_to_consider.id), best_value)

            if (index, node_to_consider.half_move, node_to_consider.id) < best_value:
                best_node = node_to_consider
                best_value = (index, node_to_consider.half_move, node_to_consider.id)

        # print('best_value',best_value)
        best_node_2 = min(nodes_to_consider_dict, key=nodes_to_consider_dict.get)  # next(iter(nodes_to_consider_dict))
        # print('~', nodes_to_consider_dict)
        print('best_node', best_node.id, best_node.index, best_node.half_move)
        # best_node.print_a_move_sequence_from_root()
        assert (best_node.half_move == self.tree.root_node.half_move + 1 + depth_picked)
        #best_node.print_a_move_sequence_from_root()
        #print('best_node2', best_node_2.id, best_node_2.index, best_node_2.half_move,
        #      self.tree.first_moves[best_node_2])

        assert (best_node == best_node_2)
        assert (not best_node.all_legal_moves_generated)
        #  self.tree.descendants.print_info()
        for nfm in self.tree.first_moves[best_node]:
            nfm.descendants_not_opened.remove_descendant(best_node)

        opening_instructions = self.opening_instructor.instructions_to_open_all_moves(best_node)
        #    opening_instructions.print_info()
        return opening_instructions

    def tree_explore(self, board):
        print(board.chess_board)
        self.count_refresh_index = 0
        self.next_update = 10

        return super().tree_explore(board)

    def print_info(self):
        super().print_info()
        print('ZipfSequool')
