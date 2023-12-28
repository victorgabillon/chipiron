from players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
from players.move_selector.treevalue.nodes.index_tree_node import IndexTreeNode
from players.move_selector.treevalue.node_selector.notations_and_statics import zipf_picks_random_bool
from players.move_selector.treevalue.nodes.tree_node_with_proportions import ProportionsNode
from players.move_selector.treevalue.trees.descendants import SortedDescendants
from players.move_selector.treevalue.nodes.tree_node_with_descendants import NodeWithDescendantsNoUpdate
from players.move_selector.treevalue.trees.first_moves import FirstMoves
import math
from players.move_selector.treevalue.node_selector.move_explorer import ProportionMoveExplorer


class ZipfSequoolTreeNode(ProportionsNode, IndexTreeNode):
    pass


class IndexDescendantsTreeNode(NodeWithDescendantsNoUpdate, IndexTreeNode):
    pass


class ZipfSequoolTree(MoveAndValueTree):

    def __init__(self, board_evaluator, board):
        super().__init__(board_evaluator, board)
        self.first_moves = FirstMoves()
        self.count_visits_at_depth = {}

    def create_tree_node(self, board, half_move, count, father_node):

        if half_move == self.tree_root_half_move:
            new_node = ZipfSequoolTreeNode(board, half_move, count, father_node)
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
            node_descendants_candidates_to_open = node.get_descendants_candidate_to_open()

            for descendant_not_opened in node_descendants_candidates_to_open:
                if not new_first_move.descendants_candidates_to_open.contains_node(descendant_not_opened):
                    index = descendant_not_opened.index if descendant_not_opened.index is not None else math.inf
                    value = (index, descendant_not_opened.half_move, descendant_not_opened.id)
                    # assert(descendant_not_opened.id !=5857)
                    new_first_move.descendants_candidates_to_open.add_descendant(descendant_not_opened, value)

    def update_after_node_creation(self, node, parent_node):

        self.first_moves.add_first_move(node, parent_node)

        node_depth = node.half_move - self.tree_root_half_move

        if node_depth == 1:
            if node not in self.count_visits_at_depth:
                self.count_visits_at_depth[node] = []

        node_depth = node.half_move - self.tree_root_half_move

        if node_depth == 1:
            node.descendants_candidates_to_open = SortedDescendants()

        if node_depth >= 1:
            if not node.is_over():
                for first_move in self.first_moves[node]:
                    first_move.descendants_candidates_to_open.add_descendant(node, (math.inf, node.half_move, node.id))

        if node_depth >= 2:  # todo can we do this strategy without the descendants at each first child?
            for first_move in self.first_moves[node]:
                first_move.descendants.add_descendant(node)

    def update_all_indices(self):
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    child.index = None  # root node is not set to zero haha

        self.root_node.index = 0
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    index = max(abs(child.get_value_white() - node.get_value_white()) / 2, node.index)
                    index = abs(
                        child.get_value_white() - node.get_value_white()) / 2 + node.index  # todo why not squared?

                    if child.index is None:
                        child.index = index
                    else:
                        child.index = min(child.index, index)
                    if not child.is_over():
                        for fm in self.first_moves[child]:
                            # print('@',self.first_moves[child])
                            if not child.all_legal_moves_generated:
                                fm.descendants_candidates_to_open.update_value(child,
                                                                               (child.index, child.half_move, child.id))

        for half_move in self.descendants:
            # print('eee',self.get_max_depth())
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    assert (child.index is not None)


class ZipfSequool:

    def __init__(self, arg, random_generator):
        super().__init__(arg)
        self.random_generator = random_generator
        self.move_explorer = ProportionMoveExplorer(arg['move_explorer_priority'])

    def create_tree(self, board):
        return ZipfSequoolTree(self.board_evaluators_wrapper, board)

    def who_is_there_smth_to_open(self, node, half_move):
        if half_move not in node.descendants_candidates_to_open:
            #  print('~~')
            return {}
        else:
            return node.descendants_candidates_to_open.sorted_descendants_at_half_move[half_move]

    def candidates_at_depth(self, node, depth_):
        root_node_half_move = self.tree.root_node.half_move
        return self.who_is_there_smth_to_open(node, 1 + root_node_half_move + depth_)  # todo this +1 is a bit confusing

    def how_many_opened_at_depth(self, node, depth_):
        root_node_half_move = self.tree.root_node.half_move
        half_move = 1 + root_node_half_move + depth_
        if half_move not in node.descendants_candidates_to_open:
            return node.descendants.number_of_descendants_at_half_move[half_move]
        else:
            return node.descendants.number_of_descendants_at_half_move[half_move]
            - node.descendants_candidates_to_open.sorted_descendants_at_half_move[half_move]

    # def test(self):
    #     for move, child in self.tree.root_node.moves_children.items():
    #         for hm in child.descendants_non_opened:
    #             for node in child.descendants_non_opened[hm]:
    #                 assert()

    def choose_node_and_move_to_open(self):
        # self.tree.update_all_indices()
        # self.tree.root_node.print_proportions()

        self.count_refresh_index += 1
        if self.tree.root_node.are_all_moves_and_children_opened():
            if self.count_refresh_index < 10 or self.count_refresh_index == self.next_update:
                if self.count_refresh_index == self.next_update:
                    self.next_update = int(1.3 * self.next_update + 1)
                pass
                # self.tree.update_all_indices()

        node = self.tree.root_node
        if node.moves_children == {}:
            return self.opening_instructor.instructions_to_open_all_moves(node)

        last_node_in_best_line = self.tree.root_node.best_node_sequence[-1]
        if last_node_in_best_line.board.is_attacked(
                not last_node_in_best_line.player_to_move) and not last_node_in_best_line.is_over():
            # print('best line is underattacked')

            if self.random_generator.random() > .5:
                # todo having to do this remove is ugle and needs to be repeated in all retrun please change!!
                for nfm in self.tree.first_moves[last_node_in_best_line]:
                    nfm.descendants_candidates_to_open.remove_descendant(last_node_in_best_line)
                return self.opening_instructor.instructions_to_open_all_moves(last_node_in_best_line)

        # node_first_move = node.choose_child_with_visits_and_proportions()
        #        node_first_move = node.choose_child_with_proportions()
        node_first_move = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=node)

        # print('firstmove', self.tree.root_node.moves_children.inverse[node_first_move])

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
        best_line_from_selected_first_move = node_first_move.best_node_sequence_not_over()
        list_elements_ = list(node_first_move.descendants.keys())[0: 1 + len(best_line_from_selected_first_move)]

        depth_picked = zipf_picks_random_bool(ordered_list_elements=list_elements_,
                                              bool_func=func_2)

        nodes_to_consider_dict = self.candidates_at_depth(node_first_move, depth_picked)
        nodes_to_consider_list = list(nodes_to_consider_dict.keys())  # todo loosing time in the list conversion?


        best_node = nodes_to_consider_list[0]
        index = best_node.index if best_node.index is not None else math.inf
        index = best_node.compute_index()

        best_value = (index, best_node.half_move, best_node.id)
        for node_to_consider in nodes_to_consider_list:
            # print('nodetoconsider',node_to_consider.id)
            index = node_to_consider.index if node_to_consider.index is not None else math.inf
            index = node_to_consider.compute_index()

            #    print('~', node_to_consider.index, (index, node_to_consider.half_move, node_to_consider.id), best_value)

            if (index, node_to_consider.half_move, node_to_consider.id) < best_value:
                best_node = node_to_consider
                best_value = (index, node_to_consider.half_move, node_to_consider.id)


        assert (best_node.half_move == self.tree.root_node.half_move + 1 + depth_picked)

        assert (not best_node.all_legal_moves_generated)
        assert (not node.is_over())
        for nfm in self.tree.first_moves[best_node]:
            nfm.descendants_candidates_to_open.remove_descendant(best_node)

        opening_instructions = self.opening_instructor.instructions_to_open_all_moves(best_node)
        #    opening_instructions.print_info()
        return opening_instructions

    def explore(self, board):
        print(board)
        self.count_refresh_index = 0
        self.next_update = 10

        return super().explore(board)

    def print_info(self):
        super().print_info()
        print('ZipfSequool')
